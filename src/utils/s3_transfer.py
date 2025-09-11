"""
S3 Transfer Utilities for RunPod Integration

Handles downloading model artifacts from RunPod S3 storage to local filesystem.
"""

import boto3
import os
from pathlib import Path
from typing import Optional
from utils.logger import logger
from dotenv import load_dotenv


def download_from_runpod_s3(s3_prefix: str, local_dir: str, bucket: str = "40ub9vhaa7") -> bool:
    """
    Download files from RunPod S3 storage to local directory.
    
    Args:
        s3_prefix: S3 prefix (directory path) to download from
        local_dir: Local directory to download files to
        bucket: S3 bucket name (default: runpod-volume)
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Get S3 credentials from environment
        access_key = os.getenv('RUNPOD_S3_ACCESS_KEY')
        secret_key = os.getenv('RUNPOD_S3_SECRET_ACCESS_KEY')
        
        if not access_key or not secret_key:
            logger.error("RunPod S3 credentials not found in environment")
            return False
        
        # Initialize S3 client for RunPod with correct endpoint and region
        endpoint_url = 'https://s3api-us-ks-2.runpod.io'
        region_name = 'us-ks-2'
        
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name
        )
        
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket)
            logger.debug(f"Bucket {bucket} accessible")
        except Exception as e:
            logger.error(f"Cannot access bucket {bucket}: {e}")
            return False
        
        logger.debug(f"Downloading from S3: s3://{bucket}/{s3_prefix} to {local_dir}")
        
        # Create local directory if it doesn't exist
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        # List objects with the given prefix
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
        
        if 'Contents' not in response:
            logger.warning(f"No objects found with prefix {s3_prefix} in bucket {bucket}")
            return False
        
        # Download each object
        downloaded_count = 0
        for obj in response['Contents']:
            s3_key = obj['Key']
            
            # Calculate local file path
            # Remove the prefix to get relative path
            relative_path = s3_key[len(s3_prefix):].lstrip('/')
            if not relative_path:  # Skip if it's just the prefix itself
                continue
                
            local_file_path = local_path / relative_path
            
            # Create parent directories if needed
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"Downloading {s3_key} to {local_file_path}")
            s3_client.download_file(bucket, s3_key, str(local_file_path))
            downloaded_count += 1
        
        logger.info(f"✅ Successfully downloaded {downloaded_count} files from s3://{bucket}/{s3_prefix} to {local_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download from S3: {e}")
        return False


def cleanup_s3_files(s3_prefix: str, bucket: str = "40ub9vhaa7") -> bool:
    """
    Clean up S3 files after successful download to save storage space.
    
    Args:
        s3_prefix: S3 prefix (directory path) to clean up
        bucket: S3 bucket name (default: runpod-volume)
        
    Returns:
        True if cleanup successful, False otherwise
    """
    try:
        # Load environment variables  
        load_dotenv()
        
        # Get S3 credentials from environment
        access_key = os.getenv('RUNPOD_S3_ACCESS_KEY')
        secret_key = os.getenv('RUNPOD_S3_SECRET_ACCESS_KEY')
        
        if not access_key or not secret_key:
            logger.error("RunPod S3 credentials not found in environment")
            return False
        
        # Initialize S3 client for RunPod with correct endpoint and region
        endpoint_url = 'https://s3api-us-ks-2.runpod.io'
        region_name = 'us-ks-2'
        
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name
        )
        
        logger.debug(f"Cleaning up S3 files with prefix: s3://{bucket}/{s3_prefix}")
        
        # List objects with the given prefix
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
        
        if 'Contents' not in response:
            logger.debug(f"No objects found to clean up with prefix {s3_prefix}")
            return True
        
        # Delete each object
        deleted_count = 0
        for obj in response['Contents']:
            s3_key = obj['Key']
            logger.debug(f"Deleting {s3_key}")
            s3_client.delete_object(Bucket=bucket, Key=s3_key)
            deleted_count += 1
        
        logger.info(f"✅ Cleaned up {deleted_count} files from s3://{bucket}/{s3_prefix}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to cleanup S3 files: {e}")
        return False


def upload_to_runpod_s3(local_dir: str, s3_prefix: str, bucket: str = "40ub9vhaa7") -> Optional[dict]:
    """
    Upload files from local directory to RunPod S3 storage.
    
    Args:
        local_dir: Local directory to upload files from
        s3_prefix: S3 prefix (directory path) to upload to
        bucket: S3 bucket name (default: runpod-volume)
        
    Returns:
        Dictionary with upload info if successful, None otherwise
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Get S3 credentials from environment
        access_key = os.getenv('RUNPOD_S3_ACCESS_KEY')
        secret_key = os.getenv('RUNPOD_S3_SECRET_ACCESS_KEY')
        
        if not access_key or not secret_key:
            logger.error("RunPod S3 credentials not found in environment")
            return None
        
        # Initialize S3 client for RunPod with correct endpoint and region
        endpoint_url = 'https://s3api-us-ks-2.runpod.io'
        region_name = 'us-ks-2'
        
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name
        )
        
        local_path = Path(local_dir)
        if not local_path.exists():
            logger.warning(f"Local directory {local_dir} does not exist")
            return None
        
        logger.debug(f"Uploading to S3: {local_dir} to s3://{bucket}/{s3_prefix}")
        
        # Upload all files in the directory recursively
        uploaded_count = 0
        uploaded_files = []
        
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                # Calculate relative path from local_dir
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')
                
                logger.debug(f"Uploading {file_path} to {s3_key}")
                s3_client.upload_file(str(file_path), bucket, s3_key)
                uploaded_count += 1
                uploaded_files.append(s3_key)
        
        if uploaded_count > 0:
            logger.info(f"✅ Successfully uploaded {uploaded_count} files from {local_dir} to s3://{bucket}/{s3_prefix}")
            
            return {
                'success': True,
                's3_prefix': s3_prefix,
                'bucket': bucket,
                'uploaded_files': uploaded_files,
                'uploaded_count': uploaded_count
            }
        else:
            logger.warning(f"No files found to upload in {local_dir}")
            return None
        
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return None