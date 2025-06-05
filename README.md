# Enhancing AI Developer Experience with Red Hat: A Practical Guide to InstructLab and RHEL AI

In today's rapidly evolving technological landscape, artificial intelligence has become a cornerstone of innovation for enterprises across industries. However, developing, deploying, and managing AI applications at scale presents significant challenges that many organizations struggle to overcome. Red Hat's comprehensive AI portfolio—including InstructLab, Red Hat Enterprise Linux (RHEL) AI, and OpenShift AI—offers a powerful solution to these challenges, enabling a seamless developer experience while maintaining enterprise-grade reliability and security.

## The Current State of AI Development Challenges

Before diving into Red Hat's solutions, let's examine the pain points that AI developers commonly face:

1. **Limited access to customizable foundation models**: Many organizations struggle to adapt general-purpose large language models (LLMs) to their specific business domains without extensive ML expertise
2. **Lengthy development cycles**: Taking AI projects from experimentation to production often involves complex handoffs between data scientists and operations teams
3. **Infrastructure complexity**: Managing the specialized hardware and software requirements for AI workloads adds significant overhead
4. **Governance and security concerns**: Ensuring models comply with organizational policies and security standards remains challenging
5. **Deployment and scaling limitations**: Many AI projects fail when transitioning from pilot to production due to scalability issues

Red Hat's AI portfolio addresses these challenges head-on, providing a comprehensive platform for developing, training, deploying, and managing AI applications across hybrid environments.

## Red Hat's AI Portfolio: An Overview

Red Hat offers a complete ecosystem for AI development:

- **InstructLab**: A community-driven project that simplifies LLM tuning and customization
- **RHEL AI**: A foundation model platform combining Granite models with InstructLab in a bootable RHEL image
- **OpenShift AI**: A comprehensive platform for managing the complete AI/ML lifecycle at scale

Let's explore how InstructLab and RHEL AI work together through a practical use case.

## Use Case: Building a Customer Support AI Assistant with Domain-Specific Knowledge

For our example, let's consider a telecommunications company that wants to create an AI-powered customer support assistant that can access and utilize their proprietary product knowledge base. This assistant needs to:

1. Handle common customer inquiries about telecom products
2. Answer questions specific to the company's services, policies, and technical specifications

### Solution Architecture

Our solution will leverage Red Hat's AI portfolio to create a comprehensive workflow:

1. Use InstructLab to customize a foundation model with telecom-specific knowledge
2. Deploy the model using RHEL AI for experimentation and initial testing
3. Scale and productionize the application with OpenShift AI

Let's break down the implementation details.

## Step 1: Customizing the Foundation Model with InstructLab

InstructLab provides a user-friendly approach to enhancing LLMs with specific domain knowledge without requiring extensive machine learning expertise.

### Installation and Setup

First, let's install and set up InstructLab with Python virtual environment:

```bash
# Create a new directory for our project
mkdir telecom-ai && cd telecom-ai

# Create a Python virtual environment
python3 -m venv --upgrade-deps venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Clear pip cache (recommended)
pip cache remove llama_cpp_python

# Install InstructLab
pip install instructlab

# Initialize InstructLab configuration
ilab config init
```

During the initialization, InstructLab will ask for the path to the taxonomy repository and model locations. You can accept the defaults by pressing Enter.

### Creating the Telecom Knowledge Taxonomy

Next, we'll create a taxonomy of telecommunications knowledge. Create the necessary directories and files:

```bash
# Create the required directory structure
mkdir -p ~/.local/share/instructlab/taxonomy/knowledge/telecom
```

Create a file called `qna.yaml` in this directory with the following content:

```yaml
version: 3
domain: telecom
document_outline: Overview of telecommunications technologies and services
created_by: mrigankapaul
seed_examples:
  - context: |
      Fiber optic internet is a broadband connection that uses fiber optic cables made of thin strands
      of glass or plastic to transmit data using light. These cables can transmit data at speeds up to
      10 Gbps or higher, making them significantly faster than traditional copper connections.
    questions_and_answers:
      - question: What is fiber optic internet?
        answer: |
          Fiber optic internet is a broadband connection that uses fiber optic cables made of thin strands
          of glass or plastic to transmit data using light signals instead of electrical signals.
      - question: What materials are fiber optic cables made from?
        answer: Fiber optic cables are made from thin strands of glass or plastic that can transmit light signals.
      - question: How fast can fiber optic internet transmit data?
        answer: |
          Fiber optic internet can transmit data at speeds up to 10 Gbps or higher, which is significantly
          faster than traditional copper-based connections.
  - context: |
      5G is the fifth generation of cellular network technology, offering significantly faster data
      transmission speeds, lower latency, and greater capacity than previous generations. 5G networks
      operate on higher frequency bands and use smaller cell sites than 4G networks.
    questions_and_answers:
      - question: What is 5G technology?
        answer: 5G is the fifth generation of cellular network technology that offers faster speeds, lower latency, and greater capacity than previous generations.
      - question: How does 5G differ from previous cellular network generations?
        answer: 5G operates on higher frequency bands, uses smaller cell sites, and provides significantly faster data transmission speeds and lower latency than 4G and earlier generations.
      - question: What are the advantages of 5G for mobile users?
        answer: 5G provides mobile users with dramatically faster download and upload speeds, more reliable connections, and the ability to connect more devices simultaneously.
  - context: |
      VoIP (Voice over Internet Protocol) is a technology that allows voice calls to be made over
      internet connections rather than traditional phone lines. It converts analog voice signals into
      digital data packets that are transmitted over IP networks.
    questions_and_answers:
      - question: What is VoIP technology?
        answer: VoIP (Voice over Internet Protocol) is a technology that allows voice calls to be made over internet connections rather than traditional phone lines.
      - question: How does VoIP work?
        answer: VoIP works by converting analog voice signals into digital data packets that are transmitted over IP networks, then converted back to voice at the receiving end.
      - question: What are the benefits of using VoIP over traditional phone services?
        answer: VoIP offers benefits such as lower costs (especially for long-distance calls), greater flexibility, enhanced features like video conferencing, and the ability to make calls from multiple devices.
  - context: |
      Satellite internet provides connectivity through communications satellites rather than land-based
      cables. It's particularly valuable for remote areas where traditional infrastructure is limited or
      absent. Modern satellite internet services can offer download speeds of up to 100 Mbps.
    questions_and_answers:
      - question: What is satellite internet?
        answer: Satellite internet is a connectivity solution that transmits data through communications satellites orbiting Earth rather than through land-based cables.
      - question: Where is satellite internet most valuable?
        answer: Satellite internet is most valuable in remote or rural areas where traditional cable or fiber infrastructure is limited or entirely absent.
      - question: What speeds can modern satellite internet achieve?
        answer: Modern satellite internet services can offer download speeds of up to 100 Mbps, though they typically have higher latency than land-based connections.
  - context: |
      Network latency refers to the time delay between when data is sent and when it's received across
      a network, typically measured in milliseconds. Low latency is crucial for applications requiring
      real-time interaction, such as video calls, online gaming, and industrial control systems.
    questions_and_answers:
      - question: What is network latency?
        answer: Network latency is the time delay between when data is sent and when it's received across a network, typically measured in milliseconds.
      - question: Why is low latency important in telecommunications?
        answer: Low latency is important for applications requiring real-time interaction, such as video calls, online gaming, financial trading, and industrial control systems.
      - question: What factors can increase network latency?
        answer: Network latency can be increased by physical distance between endpoints, network congestion, routing equipment performance, and the use of certain transmission media like satellite connections.
document:
  repo: https://github.com/instructlab/taxonomy.git
  commit: main
  patterns:
    - README.md
```

Be careful about formatting when creating this file:
- Make sure there are no trailing spaces at the end of lines
- Ensure proper indentation
- Add a newline at the end of the file

### Verify and Process the Taxonomy

Now, let's verify our taxonomy is properly formatted:

```bash
# Verify our taxonomy changes
ilab taxonomy diff
```

If all is well, you'll see a message indicating your taxonomy is valid. Next, we'll generate synthetic training data and train the model:

```bash
# Download a model to work with (if not already done during initialization)
ilab model download

# Generate synthetic training data
ilab data generate

# Train the model
ilab model train

# Convert model for macOS compatibility (required for Apple Silicon)
ilab model convert --model-dir ~/.local/share/instructlab/checkpoints/instructlab-granite-7b-lab-mlx-q
```

The training process may take some time depending on your hardware. Once complete, we can test the enhanced model.

InstructLab's approach enables us to efficiently improve the model's understanding of telecommunications concepts with minimal training data and computing resources.

### Testing Your Fine-Tuned Model

After completing the training and conversion, you can test your model using the chat interface:

```bash
ilab model chat --model ./instructlab-granite-7b-lab-trained/instructlab-granite-7b-lab-Q4_K_M.gguf
```

This starts an interactive session where you can evaluate how well your model has learned the telecom domain. Here are some effective questions to test your model with:

1. **"What is fiber optic internet and how fast can it be?"**
   - Tests if the model can combine information from multiple related question-answer pairs

2. **"How does 5G technology compare to previous generations?"**
   - Checks the model's understanding of comparative technological information

3. **"Can you explain how VoIP technology works and its advantages?"**
   - Tests if the model can synthesize technical functionality with business benefits

4. **"Why might someone in a rural area consider satellite internet?"**
   - Evaluates the model's understanding of use case scenarios for specific technologies

5. **"What telecommunications technology would be best for real-time gaming?"**
   - Application question requiring the model to consider latency requirements across technologies

6. **"What's the difference between fiber and satellite internet?"**
   - Tests cross-context learning by requiring comparison between two different technology sections

Start with questions directly from your seed examples, then gradually introduce questions requiring synthesis across multiple contexts. This progression will help you evaluate both knowledge retention and the model's ability to generalize from its training.

## Step 2: Model Deployment with RHEL AI

With our enhanced model ready, we'll deploy it using RHEL AI for initial testing and validation. RHEL AI provides a bootable image with everything needed to run and serve our model.

### Setting Up RHEL AI on AWS

For scalable deployment and testing, we'll use RHEL AI on Amazon Web Services (AWS):

#### Prerequisites

Before starting the RHEL AI installation on AWS, ensure you have:

- **Active AWS account with proper permissions**
- **Red Hat subscription** to access RHEL AI downloads
- **AWS CLI installed and configured** with your access key ID and secret access key
- **Sufficient AWS resources**: VPC, subnet, security group, and SSH key pair
- **Storage requirements**: Minimum 1TB for `/home` directory and 120GB for `/` path

#### Step 1: Install and Configure AWS CLI

If not already installed:

```bash
# Download and install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS CLI
aws configure
```

#### Step 2: Set Up Environment Variables

Create the necessary environment variables:

```bash
export BUCKET=<custom_bucket_name>
export RAW_AMI=rhel-ai-nvidia-aws-1.5-1747399384-x86_64.raw
export AMI_NAME="rhel-ai"
export DEFAULT_VOLUME_SIZE=1000  # Size in GB
```

#### Step 3: Create S3 Bucket and IAM Setup

Create an S3 bucket for image conversion:

```bash
aws s3 mb s3://$BUCKET
```

Create a trust policy file for VM import:

```bash
printf '{ 
  "Version": "2012-10-17", 
  "Statement": [ 
    { 
      "Effect": "Allow", 
      "Principal": { 
        "Service": "vmie.amazonaws.com" 
      }, 
      "Action": "sts:AssumeRole", 
      "Condition": { 
        "StringEquals":{ 
          "sts:Externalid": "vmimport" 
        } 
      } 
    } 
  ] 
}' > trust-policy.json

# Create the IAM role
aws iam create-role --role-name vmimport --assume-role-policy-document file://trust-policy.json
```

Create role policy for S3 bucket access:

```bash
printf '{
   "Version":"2012-10-17",
   "Statement":[
      {
         "Effect":"Allow",
         "Action":[
            "s3:GetBucketLocation",
            "s3:GetObject",
            "s3:ListBucket" 
         ],
         "Resource":[
            "arn:aws:s3:::%s",
            "arn:aws:s3:::%s/*"
         ]
      },
      {
         "Effect":"Allow",
         "Action":[
            "ec2:ModifySnapshotAttribute",
            "ec2:CopySnapshot",
            "ec2:RegisterImage",
            "ec2:Describe*"
         ],
         "Resource":"*"
      }
   ]
}' $BUCKET $BUCKET > role-policy.json

aws iam put-role-policy --role-name vmimport --policy-name vmimport-$BUCKET --policy-document file://role-policy.json
```

#### Step 4: Download and Convert RHEL AI Image

1. Go to the [Red Hat Enterprise Linux AI download page](https://access.redhat.com)
2. Download the RAW image file (rhel-ai-nvidia-aws-1.5-1747399384-x86_64.raw)
3. Upload it to your S3 bucket:

```bash
aws s3 cp rhel-ai-nvidia-aws-1.5-1747399384-x86_64.raw s3://$BUCKET/
```

Create the import configuration and convert the image:

```bash
# Create import configuration
printf '{ 
  "Description": "RHEL AI Image", 
  "Format": "raw", 
  "UserBucket": { 
    "S3Bucket": "%s", 
    "S3Key": "%s" 
  } 
}' $BUCKET $RAW_AMI > containers.json

# Start the import process
task_id=$(aws ec2 import-snapshot --disk-container file://containers.json | jq -r .ImportTaskId)

# Monitor import progress
aws ec2 describe-import-snapshot-tasks --filters Name=task-state,Values=active
```

Wait for the import to complete, then register the AMI:

```bash
# Get snapshot ID from completed import
snapshot_id=$(aws ec2 describe-import-snapshot-tasks --import-task-ids $task_id | jq -r '.ImportSnapshotTasks[0].SnapshotTaskDetail.SnapshotId')

# Tag the snapshot
aws ec2 create-tags --resources $snapshot_id --tags Key=Name,Value="$AMI_NAME"

# Register AMI from snapshot
ami_id=$(aws ec2 register-image \
  --name "$AMI_NAME" \
  --description "$AMI_NAME" \
  --architecture x86_64 \
  --root-device-name /dev/sda1 \
  --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=${DEFAULT_VOLUME_SIZE},SnapshotId=${snapshot_id}}" \
  --virtualization-type hvm \
  --ena-support \
  | jq -r .ImageId)

# Tag the AMI
aws ec2 create-tags --resources $ami_id --tags Key=Name,Value="$AMI_NAME"
```

#### Step 5: Launch RHEL AI Instance

Set up instance configuration variables:

```bash
instance_name=rhel-ai-instance
ami=$ami_id  # From previous step
instance_type=g4dn.xlarge  # GPU-enabled instance for AI workloads
key_name=<your-key-pair-name>
security_group=<your-sg-id>
subnet=<your-subnet-id>
disk_size=1000  # GB
```

Launch the instance:

```bash
aws ec2 run-instances \
  --image-id $ami \
  --instance-type $instance_type \
  --key-name $key_name \
  --security-group-ids $security_group \
  --subnet-id $subnet \
  --block-device-mappings DeviceName=/dev/sda1,Ebs='{VolumeSize='$disk_size'}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value='$instance_name'}]'
```

#### Step 6: Connect and Verify Installation

Connect to your instance:

```bash
ssh -i your-key.pem cloud-user@<instance-public-ip>
```

Verify RHEL AI installation:

```bash
# Verify InstructLab tools
ilab --help

# Initialize InstructLab (first time)
ilab config init
```

### Transferring Your Enhanced Model

After training a model with InstructLab on your local development machine, you need to transfer it to your RHEL AI AWS instance:

1. **Export Your Model**:
   ```bash
   # On your local development machine where you trained the model
   # Identify the model path (look for the converted GGUF model)
   ls ./instructlab-granite-7b-lab-trained/
   
   # Archive the model for transfer
   tar -czvf telecom-model.tar.gz ./instructlab-granite-7b-lab-trained/
   ```

2. **Transfer to RHEL AI AWS Instance**:
   ```bash
   # Using scp to transfer to AWS instance
   scp -i your-key.pem telecom-model.tar.gz cloud-user@<instance-public-ip>:/home/cloud-user/
   ```

3. **Extract on RHEL AI**:
   ```bash
   # SSH into your RHEL AI AWS instance
   ssh -i your-key.pem cloud-user@<instance-public-ip>
   
   # Extract the model
   mkdir -p ~/models
   tar -xzvf telecom-model.tar.gz -C ~/models
   ```

### Deploying the Model with RHEL AI Built-in Capabilities

RHEL AI comes with InstructLab pre-installed and includes support for serving models:

1. **Configure InstructLab for GGUF Models**:
   ```bash
   # Check current InstructLab configuration
   ilab config show
   
   # Edit the configuration file to use llama-cpp backend
   vi ~/.config/instructlab/config.yaml
   ```

   Update the serve section to use llama-cpp backend:
   ```yaml
   serve:
     backend: llama-cpp  # Change from vllm to llama-cpp
   ```

2. **Serve and Test the Model**:
   
   In the first terminal, start the model server:
   ```bash
   # Start the model server
   ilab model serve --model-path ~/models/instructlab-granite-7b-lab-trained/instructlab-granite-7b-lab-Q4_K_M.gguf
   ```
   
   The server will start and display messages indicating it's ready to accept connections. Keep this terminal open and running.
   
   In a second terminal, connect to the served model:
   ```bash
   # Open a new terminal and SSH into your RHEL AI instance again
   ssh -i your-key.pem cloud-user@<instance-public-ip>
   
   # Connect to the served model for interactive chat
   ilab model chat
   ```
   
   You can now interact with your fine-tuned telecom support assistant. Try asking questions like:
   - "What is fiber optic internet?"
   - "How does 5G compare to 4G?"
   - "What are the benefits of VoIP?"

### Performance Considerations for AWS RHEL AI

When running AI models on AWS RHEL AI:

1. **Instance Types**: Choose GPU-enabled instances (g4dn, p3, p4d) for optimal AI workload performance. The g4dn.xlarge instance type provides a good balance of cost and performance for testing.

2. **Storage**: RHEL AI requires minimum 1TB for `/home` directory (InstructLab data) and 120GB for `/` path (system updates). The AWS setup automatically configures appropriate storage.

3. **Security**: Configure security groups to allow necessary ports:
   - Port 22 for SSH access
   - Port 8000 for the model server (if exposing externally)

4. **Cost Management**: Monitor AWS costs as GPU instances can be expensive. Consider using spot instances for development and testing to reduce costs.

This setup provides a production-ready environment for testing your model's responses and making iterative improvements.

## Common Issues and Troubleshooting

When working with InstructLab and RHEL AI, you might encounter some common issues:

### Taxonomy Validation Errors

InstructLab has strict requirements for taxonomy files:
- **No trailing spaces**: Make sure there are no spaces at the end of lines in your YAML files
- **Taxonomy version**: Use version 3 for knowledge taxonomies
- **Required fields**: Ensure all required fields (domain, document, questions_and_answers) are present
- **Minimum examples**: Knowledge taxonomies require at least 5 seed examples
- **Repository references**: The document section must reference a valid GitHub repository

To check for these issues, run:
```bash
ilab taxonomy diff
```

If you encounter validation errors, carefully review the error messages and fix each issue.

### Environment Setup Issues

If you're missing tools like `yq`:
```bash
# For macOS
brew install yq

# For Ubuntu/Debian
sudo apt-get install yq

# For Fedora/RHEL
sudo dnf install yq
```

### Model Training Performance

For better performance during model training:
- Use GPU acceleration when available
- Start with smaller datasets for initial testing
- Consider using OpenShift AI for distributed training at scale (in future deployments)

### macOS Model Compatibility

If you encounter errors about vLLM not supporting your platform during local development, remember to convert your model to GGUF format:
```bash
ilab model convert --model-dir ~/.local/share/instructlab/checkpoints/YOUR_MODEL
```

### AWS RHEL AI Deployment Issues

Common issues when deploying on AWS:
1. **Import task fails**: Check IAM permissions and S3 bucket access
2. **AMI registration fails**: Verify snapshot completed successfully  
3. **Instance launch fails**: Check VPC, subnet, and security group configurations
4. **Connection issues**: Verify security group allows SSH (port 22) from your IP
5. **Model serving issues**: Ensure GPU drivers are properly configured and model paths are correct

### RHEL AI Model Serving Issues

If you encounter issues with model serving on RHEL AI:

1. **vLLM compatibility errors**: GGUF files are not compatible with vLLM. Always configure InstructLab to use `llama-cpp` backend for GGUF models by editing `~/.config/instructlab/config.yaml`.

2. **Model server fails to start**: Check the server logs for specific error messages. Common issues include:
   - Insufficient GPU memory
   - Port already in use
   - Model file path incorrect

## Step 3: Scaling with OpenShift AI

Once we've validated our model's performance on RHEL AI, we'll leverage OpenShift AI to productionize and scale our solution. OpenShift AI provides comprehensive tools for the entire ML lifecycle, from experimentation to production deployment at scale.

### Prerequisites for OpenShift AI

Before starting with OpenShift AI, ensure you have:

1. **OpenShift Container Platform (OCP) 4.12 or later** with:
   - Minimum 3 control plane nodes
   - Minimum 3 worker nodes (preferably with GPU support for AI workloads)
   - GPU nodes with NVIDIA GPU operator installed (for optimal performance)

2. **Red Hat Subscriptions**:
   - Valid OpenShift subscription
   - OpenShift AI subscription

3. **Required CLI Tools**:
   - `oc` (OpenShift CLI)
   - `kubectl`
   - `podman` or `docker` for container building

4. **Storage Requirements**:
   - Persistent storage provisioner (e.g., OpenShift Data Foundation, AWS EBS)
   - Minimum 100GB available storage for models

### Creating a Data Science Project

Create a new project for our AI workloads:

```bash
# Create the project
oc new-project telecom-ai-prod

# Label it for OpenShift AI
oc label namespace telecom-ai-prod opendatahub.io/dashboard=true modelmesh-enabled=true
```

### Creating and Configuring PVC for Model Storage

Create a PersistentVolumeClaim to store your model:

```bash
cat << EOF | oc apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: telecom-model-pvc
  namespace: telecom-ai-prod
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
EOF
```

Verify the PVC is bound:

```bash
oc get pvc -n telecom-ai-prod
```

### Transferring Model Files to PVC

To transfer your trained model to the PVC, we'll use a temporary pod:

1. **Create a temporary pod with the PVC mounted**:

```bash
cat << EOF | oc apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: model-upload-pod
  namespace: telecom-ai-prod
spec:
  containers:
  - name: upload-container
    image: registry.access.redhat.com/ubi9/ubi:latest
    command: ["/bin/bash", "-c", "sleep 3600"]
    volumeMounts:
    - name: model-storage
      mountPath: /models
  volumes:
  - name: model-storage
    persistentVolumeClaim:
      claimName: telecom-model-pvc
EOF
```

2. **Wait for the pod to be ready**:

```bash
oc wait --for=condition=Ready pod/model-upload-pod -n telecom-ai-prod --timeout=60s
```

3. **Copy your model files to the PVC**:

```bash
# Copy to OpenShift pod from your local machine
oc cp telecom-model.tar.gz telecom-ai-prod/model-upload-pod:/models/

# Extract in the pod
oc exec -n telecom-ai-prod model-upload-pod -- tar -xzf /models/telecom-model.tar.gz -C /models/
```

4. **Clean up the temporary pod**:

```bash
oc delete pod model-upload-pod -n telecom-ai-prod
```

### Creating a Custom Model Server Container

Since GGUF format requires special handling, we'll create a custom container that can serve GGUF models:

1. **Create a Dockerfile following Red Hat standards**:

```dockerfile
# Use Red Hat Universal Base Image
FROM registry.access.redhat.com/ubi9/python-311:latest

# Switch to root for installation
USER 0

# Install system dependencies
RUN dnf install -y \
    gcc \
    gcc-c++ \
    make \
    git \
    && dnf clean all

# Install Python dependencies
RUN pip install --no-cache-dir \
    llama-cpp-python==0.2.57 \
    flask==3.0.2 \
    gunicorn==21.2.0 \
    prometheus-client==0.19.0

# Create app directory
WORKDIR /app

# Copy the server script
COPY model_server.py /app/

# Set permissions (user 1001 already exists in UBI images)
RUN chown -R 1001:0 /app && \
    chmod -R g=u /app

# Switch to non-root user
USER 1001

# Expose ports
EXPOSE 8080 9090

# Set environment variables
ENV MODEL_PATH="/models/instructlab-granite-7b-lab-Q4_K_M.gguf"
ENV HOST="0.0.0.0"
ENV PORT="8080"

# Run the server
CMD ["python", "model_server.py"]
```

2. **Create the model server Python script**:

```python
# model_server.py
import os
import json
import logging
from flask import Flask, request, jsonify
from llama_cpp import Llama
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('model_requests_total', 'Total number of requests')
REQUEST_LATENCY = Histogram('model_request_duration_seconds', 'Request latency')
ERROR_COUNT = Counter('model_errors_total', 'Total number of errors')

app = Flask(__name__)

# Load model
MODEL_PATH = os.environ.get('MODEL_PATH', '/models/model.gguf')
logger.info(f"Loading model from {MODEL_PATH}")

try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=4,
        n_gpu_layers=-1  # Use all available GPU layers
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": MODEL_PATH})

@app.route('/ready', methods=['GET'])
def ready():
    """Readiness check endpoint"""
    return jsonify({"status": "ready"})

@app.route('/v1/completions', methods=['POST'])
def completions():
    """OpenAI-compatible completions endpoint"""
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 500)
        temperature = data.get('temperature', 0.7)
        
        # Generate completion
        response = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            echo=False
        )
        
        # Format response
        result = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "telecom-assistant",
            "choices": [{
                "text": response['choices'][0]['text'],
                "index": 0,
                "finish_reason": "stop"
            }]
        }
        
        REQUEST_LATENCY.observe(time.time() - start_time)
        return jsonify(result)
        
    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"Error generating completion: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 8080))
    app.run(host=host, port=port)
```

3. **Build the container image**:

```bash
# Create a build directory
mkdir telecom-model-server
cd telecom-model-server

# Copy the Dockerfile and model_server.py to this directory
# Then build using podman for linux/amd64 architecture
podman build --platform linux/amd64 -t telecom-model-server-amd64:latest .

# Tag for your registry
podman tag telecom-model-server-amd64:latest quay.io/<your-org>/telecom-model-server-amd64:latest

# Push to registry
podman push quay.io/<your-org>/telecom-model-server-amd64:latest
```

### Deploying with Custom Deployment

Since GGUF models require special handling with llama.cpp, we'll create a custom deployment:

1. **Create a Deployment for the model server**:

```yaml
cat << EOF | oc apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: telecom-model-server
  namespace: telecom-ai-prod
  labels:
    app: telecom-model-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: telecom-model-server
  template:
    metadata:
      labels:
        app: telecom-model-server
    spec:
      # Add tolerations to allow scheduling on GPU nodes
      tolerations:
      - key: "p4-gpu"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      containers:
      - name: model-server
        image: quay.io/<your-org>/telecom-model-server-amd64:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: MODEL_PATH
          value: "/models/instructlab-granite-7b-lab-trained/instructlab-granite-7b-lab-Q4_K_M.gguf"
        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 300
          periodSeconds: 10
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 15
          timeoutSeconds: 10
          failureThreshold: 10
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: telecom-model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: telecom-model-service
  namespace: telecom-ai-prod
spec:
  selector:
    app: telecom-model-server
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
---
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: telecom-model-route
  namespace: telecom-ai-prod
  annotations:
    haproxy.router.openshift.io/timeout: "600s"  # 10 minutes timeout for model inference
    haproxy.router.openshift.io/timeout-server: "600s"  # Server-side timeout
spec:
  to:
    kind: Service
    name: telecom-model-service
  port:
    targetPort: http
  tls:
    termination: edge
EOF
```

### Monitoring and Scaling

Set up monitoring for your deployed model:

```yaml
cat << EOF | oc apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: telecom-model-metrics
  namespace: telecom-ai-prod
spec:
  selector:
    matchLabels:
      app: telecom-model-server
  endpoints:
  - port: metrics
    interval: 30s
EOF
```

Configure horizontal pod autoscaling:

```yaml
cat << EOF | oc apply -f -
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: telecom-model-hpa
  namespace: telecom-ai-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: telecom-model-server
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF
```

### Testing Your Deployed Model

Once deployed, test your model:

```bash
# Get the route URL
MODEL_URL=$(oc get route telecom-model-route -n telecom-ai-prod -o jsonpath='{.spec.host}')

# Test the health endpoint first
curl -k https://${MODEL_URL}/health

# Start with a minimal request to verify connectivity
curl -k -X POST https://${MODEL_URL}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hi",
    "max_tokens": 10,
    "temperature": 0.7
  }'

# Once minimal requests work, test with larger requests
# Note: Larger token counts will take longer to process
curl -k -X POST https://${MODEL_URL}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is fiber optic internet?",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# For production use, consider implementing:
# 1. Streaming responses for better user experience
# 2. Request queuing for handling multiple concurrent requests
# 3. Response caching for common queries
```

### Performance Considerations

When serving GGUF models through OpenShift routes:

1. **Initial model loading**: The first request after pod startup will be slower as the model loads into memory
2. **Token generation time**: Each token takes time to generate; 200 tokens can take 30-60 seconds depending on model size and GPU
3. **Route timeouts**: Default OpenShift route timeout is 30 seconds. For LLM inference, you need longer timeouts (we set 600s)
4. **Concurrent requests**: Consider the pod's ability to handle multiple simultaneous requests

For production deployments, consider:
- Using streaming responses to provide feedback during generation
- Implementing a queue system for request management
- Setting appropriate resource limits based on your GPU capabilities

### Important Considerations for GGUF Models

1. **GGUF models require special handling**: Standard model serving frameworks expect PyTorch, TensorFlow, or ONNX formats. For GGUF, you need a custom container with llama.cpp.

2. **Performance**: GGUF models are optimized for inference and use less memory, making them ideal for edge deployments and resource-constrained environments.

3. **GPU Support**: Ensure your container has GPU support compiled in llama.cpp for optimal performance.

4. **Model Format Trade-offs**: 
   - **GGUF format** is excellent for inference-only deployments, edge computing, and scenarios where resource efficiency is critical. However, it requires custom containers and doesn't integrate with standard Kubernetes model serving APIs like KServe InferenceService.
   - **HuggingFace format** offers better operational integration with enterprise MLOps platforms, native support for KServe InferenceService APIs (enabling features like automatic scaling, canary deployments, and A/B testing), and compatibility with a wider ecosystem of tools. Consider keeping your model in HuggingFace format if you need these enterprise features and have sufficient GPU resources.

This setup provides a foundational deployment of your custom telecom AI assistant on OpenShift AI with basic monitoring and scaling capabilities. For production use, you should additionally implement:
- Authentication and authorization for API endpoints
- Network policies and security constraints
- Comprehensive logging and distributed tracing
- Model versioning and rollback strategies
- Rate limiting and circuit breakers
- Backup and disaster recovery procedures

Red Hat's InstructLab and RHEL AI provide a powerful foundation for developing and deploying custom AI applications. Through this guide, we've demonstrated how to:

1. **Customize foundation models** with domain-specific knowledge using InstructLab's intuitive taxonomy system
2. **Deploy and test models** in a production-ready RHEL AI environment on AWS
3. **Serve models efficiently** using the appropriate backend configuration for GGUF files

The telecommunications customer support example shows how these tools can be used to create practical AI solutions that incorporate proprietary knowledge. By leveraging Red Hat's open source approach, organizations can:

- Reduce dependency on generic models by adding their own domain expertise
- Accelerate AI development with user-friendly tools that don't require deep ML knowledge
- Deploy models in a secure, enterprise-ready environment
- Iterate quickly based on real-world testing and feedback

## Next Steps

To continue your AI journey with Red Hat:

1. **Expand your taxonomy**: Add more domain-specific knowledge to further enhance your model's capabilities
2. **Experiment with different base models**: Try different foundation models to see which works best for your use case
3. **Scale with OpenShift AI**: Once you've validated your approach, consider deploying at scale with OpenShift AI for production workloads
4. **Implement RAG**: Add retrieval-augmented generation to keep your model's responses current with the latest information
5. **Join the community**: Contribute to the InstructLab taxonomy repository and share your experiences with others

By mastering InstructLab and RHEL AI, you've taken the crucial first steps in building production-ready AI applications that can be customized for your specific needs while maintaining enterprise-grade reliability and security.