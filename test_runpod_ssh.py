#!/usr/bin/env python3
"""
Test SSH connectivity to RunPod serverless instance
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

def test_runpod_ssh():
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv('RUNPOD_API_KEY')
    endpoint_id = os.getenv('RUNPOD_ENDPOINT_ID')
    
    if not api_key or not endpoint_id:
        print("❌ RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID not found in environment")
        return False
    
    # Create a simple test job that starts SSH
    endpoint_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Test payload - just start SSH service and return connection info
    test_payload = {
        "input": {
            "action": "test_ssh",
            "data": {}
        }
    }
    
    print("🚀 Creating test RunPod job for SSH connectivity...")
    
    try:
        # Submit the job
        response = requests.post(endpoint_url, json=test_payload, headers=headers)
        response.raise_for_status()
        
        job_data = response.json()
        job_id = job_data.get('id')
        
        if not job_id:
            print("❌ Failed to create job")
            print(job_data)
            return False
        
        print(f"📋 Job created: {job_id}")
        print("⏳ Waiting for job to complete...")
        
        # Poll for job completion
        status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
        
        for i in range(30):  # Wait up to 5 minutes
            time.sleep(10)
            
            status_response = requests.get(status_url, headers=headers)
            status_response.raise_for_status()
            
            status_data = status_response.json()
            job_status = status_data.get('status')
            
            print(f"📊 Job status: {job_status}")
            
            if job_status == 'COMPLETED':
                output = status_data.get('output', {})
                
                if output.get('success'):
                    ssh_info = output.get('ssh_info', {})
                    print("✅ SSH test job completed successfully!")
                    print(f"🌐 SSH Host: {ssh_info.get('host', 'N/A')}")
                    print(f"🔌 SSH Port: {ssh_info.get('port', 'N/A')}")
                    print(f"👤 SSH User: {ssh_info.get('user', 'root')}")
                    
                    # Test SSH connection
                    host = ssh_info.get('host')
                    port = ssh_info.get('port', 22)
                    user = ssh_info.get('user', 'root')
                    
                    if host and port:
                        print(f"\n🔑 Testing SSH connection...")
                        ssh_command = f"ssh -o StrictHostKeyChecking=no -p {port} {user}@{host} 'echo \"SSH connection successful!\" && pwd && ls -la'"
                        print(f"Command: {ssh_command}")
                        
                        import subprocess
                        try:
                            result = subprocess.run(ssh_command, shell=True, capture_output=True, text=True, timeout=30)
                            if result.returncode == 0:
                                print("✅ SSH connection successful!")
                                print("Output:")
                                print(result.stdout)
                            else:
                                print("❌ SSH connection failed:")
                                print(result.stderr)
                        except subprocess.TimeoutExpired:
                            print("⏰ SSH connection timed out")
                        except Exception as e:
                            print(f"❌ SSH test failed: {e}")
                    
                    return True
                else:
                    error_msg = output.get('error', 'Unknown error')
                    print(f"❌ Job failed: {error_msg}")
                    return False
                    
            elif job_status in ['FAILED', 'CANCELLED', 'TIMED_OUT']:
                output = status_data.get('output', {})
                error_msg = output.get('error', f'Job {job_status.lower()}')
                print(f"❌ Job failed: {error_msg}")
                return False
        
        print("⏰ Job timed out waiting for completion")
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_runpod_ssh()
    exit(0 if success else 1)