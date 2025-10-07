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
  const backendUrl = process.env.BACKEND_INTERNAL_URL || 'http://localhost:8000';
  let port: number;
  try {
    const backendResponse = await fetch(`${backendUrl}/jobs/${jobId}/tensorboard/url`);
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

  // Use backend hostname for TensorBoard (runs on backend container)
  const backendHost = backendUrl.replace('http://', '').split(':')[0];

  // TensorBoard runs at root (no path prefix) - proxy strips /api/tensorboard/{jobId} before forwarding
  const tensorboardUrl = `http://${backendHost}:${port}/`;

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

    // Get the response body and modify HTML to add base tag for correct asset paths
    const contentType = response.headers.get('content-type') || '';
    let body: ArrayBuffer | string = await response.arrayBuffer();

    // If HTML, inject <base> tag and rewrite absolute URLs to relative
    if (contentType.includes('text/html')) {
      let html = new TextDecoder().decode(body);

      // Inject base tag right after <head> to tell browser where to resolve relative URLs
      const baseTag = `<base href="/api/tensorboard/${jobId}/">`;
      html = html.replace(/<head>/i, `<head>${baseTag}`);

      // Rewrite absolute URLs to relative (for assets that ignore <base> tag)
      // Fix: url(/font-roboto/...) -> url(font-roboto/...)
      html = html.replace(/url\(\//g, 'url(');
      // Fix: src="/index.js" -> src="index.js"
      html = html.replace(/src="\//g, 'src="');
      html = html.replace(/href="\//g, 'href="');

      console.log(`[TensorBoard Proxy Root] Injected base tag and rewrote absolute URLs for job ${jobId}`);
      body = html;
    }

    // Create response with headers to allow iframe embedding
    const nextResponse = new NextResponse(body, {
      status: response.status,
      statusText: response.statusText,
    });

    // Copy all relevant headers (skip content-length if we modified the body)
    const headersToCopy = ['content-type', 'cache-control', 'etag', 'last-modified'];
    if (!contentType.includes('text/html')) {
      headersToCopy.push('content-length'); // Only include content-length if we didn't modify the body
    }
    headersToCopy.forEach(headerName => {
      const value = response.headers.get(headerName);
      if (value) {
        nextResponse.headers.set(headerName, value);
      }
    });
    
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
    if (error instanceof Error && 'cause' in error) {
      console.error('[TensorBoard Proxy Root] Error cause:', error.cause);
    }

    return NextResponse.json(
      {
        error: 'Failed to connect to TensorBoard server',
        details: error instanceof Error ? error.message : 'Unknown error',
        cause: error instanceof Error && 'cause' in error ? String(error.cause) : undefined,
        port,
        jobId,
        suggestion: `Start TensorBoard server for job ${jobId} on port ${port}`
      },
      { status: 502 }
    );
  }
}