import boto3
import pandas as pd
import json
import io
from typing import Optional, List, Dict, Any
from botocore.exceptions import ClientError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3DatasetOrchestrator:
    """
    Helper class for orchestrating dataset operations with AWS S3
    Supports reading, writing, and managing various data formats
    """
    
    def __init__(self, aws_access_key: str, aws_secret_key: str, region: str = 'us-east-1'):
        """
        Initialize S3 client with AWS credentials
        
        Args:
            aws_access_key: AWS Access Key ID
            aws_secret_key: AWS Secret Access Key
            region: AWS region (default: us-east-1)
        """
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        self.s3_resource = boto3.resource(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        logger.info(f"S3 Orchestrator initialized for region: {region}")
    
    def list_buckets(self) -> List[str]:
        """List all available S3 buckets"""
        try:
            response = self.s3_client.list_buckets()
            buckets = [bucket['Name'] for bucket in response['Buckets']]
            logger.info(f"Found {len(buckets)} buckets")
            return buckets
        except ClientError as e:
            logger.error(f"Error listing buckets: {e}")
            raise
    
    def list_files(self, bucket: str, prefix: str = '', suffix: str = '') -> List[Dict[str, Any]]:
        """
        List files in S3 bucket with optional filtering
        
        Args:
            bucket: S3 bucket name
            prefix: Filter by prefix (folder path)
            suffix: Filter by suffix (file extension)
            
        Returns:
            List of file metadata dictionaries
        """
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            files = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if suffix == '' or obj['Key'].endswith(suffix):
                            files.append({
                                'key': obj['Key'],
                                'size': obj['Size'],
                                'last_modified': obj['LastModified'].isoformat(),
                                'storage_class': obj.get('StorageClass', 'STANDARD')
                            })
            
            logger.info(f"Found {len(files)} files in {bucket}/{prefix}")
            return files
        except ClientError as e:
            logger.error(f"Error listing files: {e}")
            raise
    
    def read_csv(self, bucket: str, key: str, **kwargs) -> pd.DataFrame:
        """
        Read CSV file from S3 into pandas DataFrame
        
        Args:
            bucket: S3 bucket name
            key: S3 object key (file path)
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            pandas DataFrame
        """
        try:
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()), **kwargs)
            logger.info(f"Read CSV: {key} - Shape: {df.shape}")
            return df
        except ClientError as e:
            logger.error(f"Error reading CSV {key}: {e}")
            raise
    
    def read_json(self, bucket: str, key: str) -> Dict:
        """
        Read JSON file from S3
        
        Args:
            bucket: S3 bucket name
            key: S3 object key (file path)
            
        Returns:
            Parsed JSON as dictionary
        """
        try:
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj['Body'].read().decode('utf-8'))
            logger.info(f"Read JSON: {key}")
            return data
        except ClientError as e:
            logger.error(f"Error reading JSON {key}: {e}")
            raise
    
    def read_parquet(self, bucket: str, key: str, **kwargs) -> pd.DataFrame:
        """
        Read Parquet file from S3 into pandas DataFrame
        
        Args:
            bucket: S3 bucket name
            key: S3 object key (file path)
            **kwargs: Additional arguments for pd.read_parquet
            
        Returns:
            pandas DataFrame
        """
        try:
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            df = pd.read_parquet(io.BytesIO(obj['Body'].read()), **kwargs)
            logger.info(f"Read Parquet: {key} - Shape: {df.shape}")
            return df
        except ClientError as e:
            logger.error(f"Error reading Parquet {key}: {e}")
            raise
    
    def read_text(self, bucket: str, key: str) -> str:
        """
        Read text file from S3
        
        Args:
            bucket: S3 bucket name
            key: S3 object key (file path)
            
        Returns:
            File content as string
        """
        try:
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = obj['Body'].read().decode('utf-8')
            logger.info(f"Read text file: {key} - Size: {len(content)} chars")
            return content
        except ClientError as e:
            logger.error(f"Error reading text {key}: {e}")
            raise
    
    def write_csv(self, df: pd.DataFrame, bucket: str, key: str, **kwargs) -> bool:
        """
        Write pandas DataFrame to S3 as CSV
        
        Args:
            df: pandas DataFrame to write
            bucket: S3 bucket name
            key: S3 object key (file path)
            **kwargs: Additional arguments for df.to_csv
            
        Returns:
            True if successful
        """
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, **kwargs)
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=csv_buffer.getvalue()
            )
            logger.info(f"Written CSV: {key} - Shape: {df.shape}")
            return True
        except ClientError as e:
            logger.error(f"Error writing CSV {key}: {e}")
            raise
    
    def write_json(self, data: Dict, bucket: str, key: str) -> bool:
        """
        Write dictionary to S3 as JSON
        
        Args:
            data: Dictionary to write
            bucket: S3 bucket name
            key: S3 object key (file path)
            
        Returns:
            True if successful
        """
        try:
            json_data = json.dumps(data, indent=2)
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json_data
            )
            logger.info(f"Written JSON: {key}")
            return True
        except ClientError as e:
            logger.error(f"Error writing JSON {key}: {e}")
            raise
    
    def write_parquet(self, df: pd.DataFrame, bucket: str, key: str, **kwargs) -> bool:
        """
        Write pandas DataFrame to S3 as Parquet
        
        Args:
            df: pandas DataFrame to write
            bucket: S3 bucket name
            key: S3 object key (file path)
            **kwargs: Additional arguments for df.to_parquet
            
        Returns:
            True if successful
        """
        try:
            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, index=False, **kwargs)
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=parquet_buffer.getvalue()
            )
            logger.info(f"Written Parquet: {key} - Shape: {df.shape}")
            return True
        except ClientError as e:
            logger.error(f"Error writing Parquet {key}: {e}")
            raise
    
    def delete_file(self, bucket: str, key: str) -> bool:
        """
        Delete file from S3
        
        Args:
            bucket: S3 bucket name
            key: S3 object key (file path)
            
        Returns:
            True if successful
        """
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=key)
            logger.info(f"Deleted: {key}")
            return True
        except ClientError as e:
            logger.error(f"Error deleting {key}: {e}")
            raise
    
    def copy_file(self, source_bucket: str, source_key: str, 
                  dest_bucket: str, dest_key: str) -> bool:
        """
        Copy file within S3 or between buckets
        
        Args:
            source_bucket: Source bucket name
            source_key: Source object key
            dest_bucket: Destination bucket name
            dest_key: Destination object key
            
        Returns:
            True if successful
        """
        try:
            copy_source = {'Bucket': source_bucket, 'Key': source_key}
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=dest_bucket,
                Key=dest_key
            )
            logger.info(f"Copied: {source_key} -> {dest_key}")
            return True
        except ClientError as e:
            logger.error(f"Error copying file: {e}")
            raise
    
    def file_exists(self, bucket: str, key: str) -> bool:
        """
        Check if file exists in S3
        
        Args:
            bucket: S3 bucket name
            key: S3 object key (file path)
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise
    
    def get_file_metadata(self, bucket: str, key: str) -> Dict[str, Any]:
        """
        Get metadata for a file in S3
        
        Args:
            bucket: S3 bucket name
            key: S3 object key (file path)
            
        Returns:
            Dictionary with file metadata
        """
        try:
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            metadata = {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'].isoformat(),
                'content_type': response.get('ContentType', 'unknown'),
                'etag': response['ETag'],
                'metadata': response.get('Metadata', {})
            }
            logger.info(f"Retrieved metadata for: {key}")
            return metadata
        except ClientError as e:
            logger.error(f"Error getting metadata for {key}: {e}")
            raise
    
    def batch_read_csvs(self, bucket: str, prefix: str = '') -> pd.DataFrame:
        """
        Read multiple CSV files and concatenate into single DataFrame
        
        Args:
            bucket: S3 bucket name
            prefix: Prefix to filter files (folder path)
            
        Returns:
            Concatenated pandas DataFrame
        """
        files = self.list_files(bucket, prefix, suffix='.csv')
        dfs = []
        
        for file_info in files:
            try:
                df = self.read_csv(bucket, file_info['key'])
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Skipping {file_info['key']}: {e}")
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined {len(dfs)} CSV files - Final shape: {combined_df.shape}")
            return combined_df
        else:
            logger.warning("No CSV files found to combine")
            return pd.DataFrame()
    
    def download_file(self, bucket: str, key: str, local_path: str) -> bool:
        """
        Download file from S3 to local filesystem
        
        Args:
            bucket: S3 bucket name
            key: S3 object key (file path)
            local_path: Local file path to save to
            
        Returns:
            True if successful
        """
        try:
            self.s3_client.download_file(bucket, key, local_path)
            logger.info(f"Downloaded: {key} -> {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Error downloading {key}: {e}")
            raise
    
    def upload_file(self, local_path: str, bucket: str, key: str) -> bool:
        """
        Upload file from local filesystem to S3
        
        Args:
            local_path: Local file path to upload
            bucket: S3 bucket name
            key: S3 object key (destination path)
            
        Returns:
            True if successful
        """
        try:
            self.s3_client.upload_file(local_path, bucket, key)
            logger.info(f"Uploaded: {local_path} -> {key}")
            return True
        except ClientError as e:
            logger.error(f"Error uploading {local_path}: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize orchestrator
    orchestrator = S3DatasetOrchestrator(
        aws_access_key='YOUR_ACCESS_KEY',
        aws_secret_key='YOUR_SECRET_KEY',
        region='us-east-1'
    )
    
    # List buckets
    buckets = orchestrator.list_buckets()
    print(f"Available buckets: {buckets}")
    
    # List files in a bucket
    files = orchestrator.list_files('my-bucket', prefix='data/', suffix='.csv')
    print(f"Found {len(files)} CSV files")
    
    # Read CSV from S3
    df = orchestrator.read_csv('my-bucket', 'data/dataset.csv')
    print(f"DataFrame shape: {df.shape}")
    
    # Write CSV to S3
    orchestrator.write_csv(df, 'my-bucket', 'output/processed_data.csv')
    
    # Read multiple CSVs and combine
    combined_df = orchestrator.batch_read_csvs('my-bucket', prefix='data/')
    print(f"Combined DataFrame shape: {combined_df.shape}")
    
    # Check if file exists
    exists = orchestrator.file_exists('my-bucket', 'data/dataset.csv')
    print(f"File exists: {exists}")
    
    # Get file metadata
    metadata = orchestrator.get_file_metadata('my-bucket', 'data/dataset.csv')
    print(f"File metadata: {metadata}")
