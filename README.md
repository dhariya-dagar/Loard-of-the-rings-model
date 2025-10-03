# Lord of the Rings Chatbot

This is a Streamlit-based chatbot that answers questions about the "Lord of the Rings". It uses a local ChromaDB vector store for retrieval, and also leverages Wikipedia and Brave Search for additional information.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up environment variables:**
    Create a `.env` file in the root of the project and add the following environment variables:
    ```
    OPENAI_API_KEY="your-openai-api-key"
    BRAIN_API_KEY="your-brave-api-key"
    GOOGLE_API_KEY="your-google-api-key"
    ```

3.  **Load the data:**
    Run the `loader.py` script to load the "Lord of the Rings" text into the ChromaDB vector store:
    ```bash
    python loader.py
    ```

## Run the application

To run the Streamlit application, use the following command:

```bash
streamlit run app.py
```

This will open the chatbot interface in your browser.
