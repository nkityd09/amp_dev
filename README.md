# 2023 MBO Demo Repository
Git Repo for CML Code Chatbot

## Key Features

- Chat with Codellama-13B model and ask code related questions


## Resource Requirements

The AMP Application has been configured to use the following 
- 4 CPU
- 32 GB RAM
- 1 GPU

## Steps to Setup the CML AMP Application

1. Navigate to CML Workspace -> Site Administration -> AMPs Tab

2. Under AMP Catalog Sources section, We will "Add a new source By" selecting "Catalog File URL" 

3. Provide the following URL and click "Add Source"

```
https://raw.githubusercontent.com/nkityd09/amp_testing/main/catalog.yaml
```

4. Click on the AMP and "Configure Project", disable Spark as it is not required.

5. Once the AMP steps are completed, We can access the Gradio UI via the Applications page.

**Note**: The application creates a "default" collection in the VectorDB when the AMP is launched the first time.

