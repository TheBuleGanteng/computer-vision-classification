import { NextRequest, NextResponse } from 'next/server';

interface Params {
  jobId: string;
  path: string[];
}

// Helper function to get TensorBoard port for job
function getTensorBoardPort(jobId: string): number {
  // Same logic as backend - consistent port allocation
  return 6006 + (Math.abs(hashCode(jobId)) % 1000);
}

function hashCode(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return hash;
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<Params> }
) {
  const { jobId, path } = await params;
  
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

  // Construct TensorBoard URL
  const pathString = path ? path.join('/') : '';
  // Use backend hostname for TensorBoard (runs on backend container)
  const backendHost = backendUrl.replace('http://', '').split(':')[0];
  const tensorboardUrl = `http://${backendHost}:${port}/${pathString}`;
  
  // Forward query parameters
  const url = new URL(request.url);
  const searchParams = url.searchParams.toString();
  const fullUrl = searchParams ? `${tensorboardUrl}?${searchParams}` : tensorboardUrl;
  
  try {
    console.log(`[TensorBoard Proxy] Forwarding request to: ${fullUrl}`);
    
    const response = await fetch(fullUrl, {
      method: 'GET',
      headers: {
        // Forward relevant headers but exclude problematic ones
        'User-Agent': request.headers.get('user-agent') || 'Next.js Proxy',
        'Accept': request.headers.get('accept') || '*/*',
        'Accept-Language': request.headers.get('accept-language') || 'en-US,en;q=0.9',
      },
    });

    if (!response.ok) {
      console.error(`[TensorBoard Proxy] Error ${response.status}: ${response.statusText}`);
      return NextResponse.json(
        { error: `TensorBoard server error: ${response.status} ${response.statusText}` },
        { status: response.status }
      );
    }

    // Get the response body
    const body = await response.arrayBuffer();
    
    // Check if it's HTML content and inject base URL
    const responseContentType = response.headers.get('content-type') || 'text/html';
    let processedBody = body;
    
    console.log(`[TensorBoard Proxy] Path: ${pathString}, Content-Type: ${responseContentType}`);
    
    // Only inject base tag for actual HTML files, not JS files
    const shouldInjectBase = responseContentType.includes('text/html') && !pathString.endsWith('.js') && !responseContentType.includes('javascript');
    console.log(`[TensorBoard Proxy] Should inject base: ${shouldInjectBase}`);
    
    if (shouldInjectBase) {
      const htmlText = new TextDecoder().decode(body);
      // Inject base tag to fix relative URLs
      const baseTag = `<base href="/api/tensorboard/${jobId}/">`;
      const modifiedHtml = htmlText.replace('<head>', `<head>\n  ${baseTag}`);
      processedBody = new TextEncoder().encode(modifiedHtml).buffer;
      console.log(`[TensorBoard Proxy] Injected base tag for HTML content`);
    } else {
      console.log(`[TensorBoard Proxy] Skipping base tag injection for ${pathString}`);
    }
    
    // Create response with modified headers to allow iframe embedding
    const nextResponse = new NextResponse(processedBody, {
      status: response.status,
      statusText: response.statusText,
    });
    
    // Copy safe headers from TensorBoard response
    const safeHeaders = ['content-type', 'content-length', 'cache-control', 'etag', 'last-modified'];
    safeHeaders.forEach(header => {
      const value = response.headers.get(header);
      if (value) {
        nextResponse.headers.set(header, value);
      }
    });
    
    // Override security headers to allow iframe embedding
    nextResponse.headers.delete('x-frame-options');
    nextResponse.headers.delete('content-security-policy');
    nextResponse.headers.set('X-Frame-Options', 'ALLOWALL');
    
    // Add CORS headers for cross-origin requests
    nextResponse.headers.set('Access-Control-Allow-Origin', '*');
    nextResponse.headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    nextResponse.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    
    return nextResponse;
    
  } catch (error) {
    console.error('[TensorBoard Proxy] Fetch error:', error);
    
    return NextResponse.json(
      { 
        error: 'Failed to connect to TensorBoard server',
        details: error instanceof Error ? error.message : 'Unknown error',
        tensorboardUrl: fullUrl,
        suggestion: `Make sure TensorBoard server is running on port ${port}`
      },
      { status: 502 }
    );
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<Params> }
) {
  const { jobId, path } = await params;
  
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

  const pathString = path ? path.join('/') : '';
  // Use backend hostname for TensorBoard (runs on backend container)
  const backendHost = backendUrl.replace('http://', '').split(':')[0];
  const tensorboardUrl = `http://${backendHost}:${port}/${pathString}`;
  
  try {
    const body = await request.arrayBuffer();
    
    const response = await fetch(tensorboardUrl, {
      method: 'POST',
      headers: {
        'Content-Type': request.headers.get('content-type') || 'application/json',
        'User-Agent': 'Next.js Proxy',
      },
      body,
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: `TensorBoard server error: ${response.status} ${response.statusText}` },
        { status: response.status }
      );
    }

    const responseBody = await response.arrayBuffer();
    const nextResponse = new NextResponse(responseBody, {
      status: response.status,
      headers: {
        'Content-Type': response.headers.get('content-type') || 'application/json',
        'Access-Control-Allow-Origin': '*',
      },
    });

    return nextResponse;
    
  } catch (error) {
    console.error('[TensorBoard Proxy] POST error:', error);
    return NextResponse.json(
      { error: 'Failed to forward POST request to TensorBoard' },
      { status: 502 }
    );
  }
}

// Handle OPTIONS requests for CORS preflight
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  });
}