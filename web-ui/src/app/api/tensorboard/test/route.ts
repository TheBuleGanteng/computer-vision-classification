import { NextResponse } from 'next/server';

// Simple test endpoint to verify proxy routing is working
export async function GET() {
  return NextResponse.json({
    message: 'TensorBoard proxy API routes are working!',
    timestamp: new Date().toISOString(),
    endpoints: [
      '/api/tensorboard/test',
      '/api/tensorboard/[jobId]',
      '/api/tensorboard/[jobId]/[...path]'
    ]
  });
}