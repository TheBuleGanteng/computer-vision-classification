import { NextRequest, NextResponse } from 'next/server';

interface Params {
  jobId: string;
}

// Helper function to get TensorBoard port for job (same as in [...path]/route.ts)
function getTensorBoardPort(jobId: string): number {
  return 6006 + (Math.abs(hashCode(jobId)) % 1000);
}

function hashCode(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return hash;
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<Params> }
) {
  const { jobId } = await params;
  
  // Get the actual TensorBoard port from the backend instead of calculating
  let port: number;
  try {
    const backendResponse = await fetch(`http://localhost:8000/jobs/${jobId}/tensorboard/url`);
    if (backendResponse.ok) {
      const backendData = await backendResponse.json();
      port = backendData.port;
    } else {
      port = getTensorBoardPort(jobId); // Fallback to calculation
    }
  } catch {
    port = getTensorBoardPort(jobId); // Fallback to calculation
  }
  
  // Root TensorBoard URL - only handle actual job IDs, not file paths
  if (jobId.includes('.') || jobId.includes('/')) {
    return NextResponse.json(
      { error: 'Invalid job ID format' },
      { status: 400 }
    );
  }
  
  const tensorboardUrl = `http://localhost:${port}/`;
  
  // Forward query parameters
  const url = new URL(request.url);
  const searchParams = url.searchParams.toString();
  const fullUrl = searchParams ? `${tensorboardUrl}?${searchParams}` : tensorboardUrl;
  
  try {
    console.log(`[TensorBoard Proxy Root] Forwarding request to: ${fullUrl}`);
    
    const response = await fetch(fullUrl, {
      method: 'GET',
      headers: {
        'User-Agent': request.headers.get('user-agent') || 'Next.js Proxy',
        'Accept': request.headers.get('accept') || 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': request.headers.get('accept-language') || 'en-US,en;q=0.9',
      },
    });

    if (!response.ok) {
      console.error(`[TensorBoard Proxy Root] Error ${response.status}: ${response.statusText}`);
      return NextResponse.json(
        { 
          error: `TensorBoard server not available: ${response.status} ${response.statusText}`,
          port,
          suggestion: 'Make sure TensorBoard server is started for this job'
        },
        { status: response.status }
      );
    }

    // Get the response body
    const body = await response.arrayBuffer();
    
    // Check if it's HTML content and inject base URL
    const responseContentType = response.headers.get('content-type') || '';
    let processedBody = body;
    
    if (responseContentType.includes('text/html')) {
      const htmlText = new TextDecoder().decode(body);
      // Inject base tag to fix relative URLs
      const baseTag = `<base href="/api/tensorboard/${jobId}/">`;
      const modifiedHtml = htmlText.replace('<head>', `<head>\n  ${baseTag}`);
      processedBody = new TextEncoder().encode(modifiedHtml).buffer;
    }
    
    // Create response with headers to allow iframe embedding
    const nextResponse = new NextResponse(processedBody, {
      status: response.status,
      statusText: response.statusText,
    });
    
    // Copy content headers
    const contentType = response.headers.get('content-type');
    if (contentType) {
      nextResponse.headers.set('content-type', contentType);
    }
    
    // Override security headers to allow iframe embedding
    nextResponse.headers.delete('x-frame-options');
    nextResponse.headers.delete('content-security-policy');
    nextResponse.headers.set('X-Frame-Options', 'ALLOWALL');
    
    // Add CORS headers
    nextResponse.headers.set('Access-Control-Allow-Origin', '*');
    nextResponse.headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    nextResponse.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    
    return nextResponse;
    
  } catch (error: unknown) {
    console.error('[TensorBoard Proxy Root] Fetch error:', error);
    
    return NextResponse.json(
      { 
        error: 'Failed to connect to TensorBoard server',
        details: error instanceof Error ? error.message : 'Unknown error',
        port,
        jobId,
        suggestion: `Start TensorBoard server for job ${jobId} on port ${port}`
      },
      { status: 502 }
    );
  }
}