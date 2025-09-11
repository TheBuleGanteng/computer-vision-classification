# Serverless storage

> Explore storage options for your Serverless endpoints, including container volumes, network volumes, and S3-compatible storage.

This guide explains the different types of storage available in Runpod Serverless, their characteristics, and when to use each option.

## Storage types

### Container volume

A worker's container volume holds temporary storage that exists only while a worker is running, and is completely lost when the worker is stopped or scaled down. It's created automatically when a Serverless worker launches and remains tightly coupled with the worker's lifecycle.

Container volumes provide fast read and write speeds since they are locally attached to workers. The cost of storage is included in the worker's running cost, making it an economical choice for temporary data.

Any data saved by a worker's handler function will be stored in the container volume by default. To persist data beyond the current worker session, use a network volume or S3-compatible storage.

### Network volume

Network volumes provide persistent storage that can be attached to different workers and even shared between multiple workers. Network volumes are ideal for sharing datasets between workers, storing large models that need to be accessed by multiple workers, and preserving data that needs to outlive any individual worker.

To learn how to create and use network volumes, see [Network volumes](/serverless/storage/network-volumes).

### S3-compatible storage integration

<Tip>
  Runpod's S3 integration works with any S3-compatible storage provider, not just AWS S3. You can use MinIO, Backblaze B2, DigitalOcean Spaces, and other compatible providers.
</Tip>

Runpod's S3-compatible storage integration allows you to connect your Serverless endpoints to external object storage services, giving you the flexibility to use your own storage provider with standardized access protocols.

You can supply your own credentials for any S3-compatible storage service, which is particularly useful for handling large files that exceed API payload limits. This storage option exists entirely outside the Runpod infrastructure, giving you complete control over data lifecycle and retention policies. Billing depends on your chosen provider's pricing model rather than Runpod's storage rates.

To configure requests to send data to S3-compatible storage, see [S3-compatible storage integration](/serverless/endpoints/send-requests#s3-compatible-storage-integration).

## Storage comparison table

| Feature         | Container Volume                     | Network Volume                      | S3-Compatible Storage             |
| --------------- | ------------------------------------ | ----------------------------------- | --------------------------------- |
| **Persistence** | Temporary (erased when worker stops) | Permanent (independent of workers)  | Permanent (external to Runpod)    |
| **Sharing**     | Not shareable                        | Can be attached to multiple workers | Accessible via S3 credentials     |
| **Speed**       | Fastest (local)                      | Fast (networked NVME)               | Varies by provider                |
| **Cost**        | Included in worker cost              | \$0.05-0.07/GB/month                | Varies by provider                |
| **Size limits** | Varies by worker config              | Up to 4TB self-service              | Varies by provider                |
| **Best for**    | Temporary processing                 | Multi-worker sharing                | Very large files, external access |

## Serverless storage behavior

### Data isolation and sharing

Each worker has its own local directory and maintains its own data. This means that different workers running on your endpoint cannot share data directly between each other (unless a network volume is attached).

### Caching and cold starts

Serverless workers cache and load their Docker images locally on the container volume, even if a network volume is attached. While this local caching speeds up initial worker startup, loading large models into GPU memory can still significantly impact cold start times.

For guidance on optimizing storage to reduce cold start times, see [Endpoint configuration](/serverless/endpoints/endpoint-configurations#reducing-worker-startup-times).

### Location constraints

If you use network volumes with your Serverless endpoint, your deployments will be constrained to the data center where the volume is located. This constraint may impact GPU availability and failover options, as your workloads must run in proximity to your storage. For global deployments, consider how storage location might affect your overall system architecture.


# Network volumes for Serverless

> Persistent, portable storage for Serverless workers.

Network volumes offer persistent storage that exists independently of the lifecycle of a [Serverless worker](/serverless/workers/overview). This means your data will be retained even if a worker is stopped or your endpoint is deleted.

Network volumes can be attached to multiple Serverless endpoints, making them ideal for sharing data, or maintaining datasets between workers.

When attached to a Serverless endpoint, a network volume is mounted at `/runpod-volume` within the worker environment.

Network volumes are billed hourly at a rate of \$0.07 per GB per month for the first 1TB, and \$0.05 per GB per month for additional storage beyond that.

<Warning>
  If your account lacks sufficient funds to cover storage costs, your network volume may be terminated. Once terminated, the disk space is immediately freed for other users, and Runpod cannot recover lost data. Ensure your account remains funded to prevent data loss.
</Warning>

## When to use a network volume

Consider using a network volume when your endpoints needs:

* **Persistent data that outlives individual workers:** Keep your data safe and accessible even after a worker is stopped or an endpoint is scaled to zero.
* **Shareable storage:** Share data between workers on the same endpoint, or across multiple endpoints.
* **Efficient data management:** Store frequently used models or large datasets to avoid re-downloading them for each new worker, saving time, bandwidth, and reducing cold start times.
* **Stateful applications:** Maintain state across multiple invocations of a Serverless function, enabling more complex, long-running tasks.

## Create a network volume

<Tabs>
  <Tab title="Web">
    To create a new network volume:

    1. Navigate to the [Storage page](https://www.console.runpod.io/user/storage) in the Runpod console.

    2. Select **New Network Volume**.

    3. **Configure your volume:**

       * Select a datacenter for your volume. Datacenter location does not affect pricing, but the datacenter location will determine which endpoints your network volume can be paired with. Your Serverless endpoint must be in the same datacenter as the network volume.
       * Provide a descriptive name for your volume (e.g., "serverless-shared-models" or "project-gamma-datasets").
       * Specify the desired size for the volume in gigabytes (GB).

       <Warning>
         Network volume size can be increased later, but cannot be decreased.
       </Warning>

    4. Select **Create Network Volume**.

    You can edit and delete your network volumes using the [Storage page](https://www.console.runpod.io/user/storage).
  </Tab>

  <Tab title="REST API">
    To create a network volume using the REST API, send a POST request to the `/networkvolumes` endpoint:

    ```bash
    curl --request POST \
      --url https://rest.runpod.io/v1/networkvolumes \
      --header 'Authorization: Bearer RUNPOD_API_KEY' \
      --header 'Content-Type: application/json' \
      --data '{
      "name": "my-network-volume",
      "size": 100,
      "dataCenterId": "EU-RO-1"
    }'
    ```

    For complete API documentation and parameter details, see the [network volumes API reference](/api-reference/network-volumes/POST/networkvolumes).
  </Tab>
</Tabs>

## Attach a network volume to an endpoint

To enable workers on an endpoint to use a network volume:

1. Navigate to the [Serverless page](https://www.console.runpod.io/serverless/user/endpoints) in the Runpod console.
2. Select an existing endpoint and click **Manage**, then select **Edit Endpoint**.
3. In the endpoint configuration menu, scroll down and expand the **Advanced** section.
4. Click **Network Volume** and select the network volume you want to attach to the endpoint.
5. Configure any other fields as you normally would, then select **Save Endpoint**.

Data from the network volume will be accessible to all workers for that endpoint from the `/runpod-volume` directory. Use this path to read and write shared data in your [handler function](/serverless/workers/handler-functions).

<Warning>
  Writing to the same network volume from multiple endpoints/workers simultaneously may result in conflicts or data corruption. Ensure your application logic handles concurrent access appropriately for write operations.
</Warning>

## S3-compatible API

Runpod provides an S3-compatible API that allows you to access and manage files on your network volumes directly, without needing to launch a Pod or run a Serverless worker. This offers several key benefits:

* **Direct file management**: Upload, download, list, and delete files on your network volumes using standard S3 operations.
* **Cost efficiency**: Manage your data without needing to launch a Pod for file management and incurring compute costs.
* **Improved cold start performance**: Pre-populate network volumes with models and datasets to reduce worker initialization time.
* **Automation-friendly**: Integrate with existing S3-compatible tools, scripts, and CI/CD pipelines.
* **Standard tooling support**: Use familiar tools like AWS CLI or Boto3 (Python).

For detailed setup instructions, authentication, and comprehensive usage examples, see the [S3-compatible API documentation](/serverless/storage/s3-api).

<Note>
  The S3-compatible API is currently available for network volumes in the following datacenters: `EUR-IS-1`, `EU-RO-1`, `EU-CZ-1`, `US-KS-2`, `US-CA-2`.
</Note>

## Architecture details

Network volumes are backed by high-performance storage servers co-located with Runpod GPU servers. These are connected via high-speed networks and use NVMe SSDs for optimal performance, but data transfer speeds can vary widely based on location and network conditions (200-400MB/s, up to 10GB/s).

## Potential benefits

Using network volumes with Serverless provides significant flexibility and can lead to improved performance and cost savings:

* **Reduced cold starts:** By storing large models or datasets on a network volume, workers can access them quickly without needing to download them on each cold start.
* **Cost efficiency:** Network volume storage space costs less than frequently re-downloading large files or relying solely on container storage for data that needs to persist.
* **Simplified data management:** Centralize your datasets and models for easier updates and management across multiple workers and endpoints.


curl --request POST \
  --url https://rest.runpod.io/v1/networkvolumes \
  --header 'Authorization: Bearer RUNPOD_API_KEY' \
  --header 'Content-Type: application/json' \
  --data '{
  "name": "my-network-volume",
  "size": 100,
  "dataCenterId": "EU-RO-1"
}'