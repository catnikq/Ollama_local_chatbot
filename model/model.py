from langchain_ollama import ChatOllama



class LLM:
    def __init__(self, model_name='Llama3.1'):
        """
        Initialize the LLM model with the specified name.
        
        Parameters:
            model_name (str): Name of the model to use. Defaults to 'Llama3.1'.
        """
        # Initialize model
        self.model = ChatOllama(model = model_name, temperature=0.3)

                
    def generate_response(self, user_input):
        """
        Generate a response from the LLM based on user input.
        
        Parameters:
            user_input (str): The input text from the user.
        
        Returns:
            response (str): The generated response from the LLM.
        """        
        # Use the model to generate a response
        response = self.model.invoke(user_input)
        
        # Extract only the text portion of the response
        # This assumes `response` is a dictionary-like object. Adjust accordingly if not.
        if isinstance(response, dict) and 'text' in response:
            return response['text']
        else:
            return str(response)  # Fallback to returning the entire response if the expected structure is different