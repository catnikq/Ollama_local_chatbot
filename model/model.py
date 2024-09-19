from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder


class LLM:
    def __init__(self, model_name='Llama3.1'):
        """
        Initialize the LLM model with the specified name.
        
        Parameters:
            model_name (str): Name of the model to use. Defaults to 'Llama3.1'.
        """
        # Initialize model
        self.model = ChatOllama(model = model_name, temperature=0.3)
        
        
    def default_prompt_template(self, user_input):
        """Call this function to use default prompt template. Else create a new prompt.
        
        Args:
            user_input(str): user input.
        
        Returns:
            default_prompt(f_string)
        """
        # Prompt template
        self.prompt_template = """
        You are an assistant. Your job is to answer user's concerns about specific documents.
        You can hold a friendly conversation but do not answer irrelevant topics that is not related to the documents.
        When you are asked with unrelated topics, you should not answer such question but offer to help with the related topic instead.
        Don't invent answers that you don't know about.
        """
        
        # Define default prompt template
        self.default_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", self.prompt_template),
            # MessagesPlaceholder("historical_context"),
            ("human", "{user_input}")
        ]
    )
        return self.default_prompt
        
        
    def create_chain(self, prompt=None, historical_context=None):
        """Create a new chain for invoking.

        Args:
            prompt (f_string, optional): Use a new prompt. Default None whil use default_prompt instead.
            historical_context (list, optional): Use chat history as context. Default None

        Returns:
            chat_chain: chain for invoking responses.
        """
        # Use the provided prompt template or default one if not specified
        prompt = prompt or self.default_prompt
        
        # Create and return a new LangChain with the LLM and specified prompt template
        chat_chain = prompt | self.model
        return chat_chain
        
        
    def generate_response(self, chain, user_input, historical_context=None):
        """
        Generate a response from the LLM based on user input.
        
        Parameters:
            user_input (str): The input text from the user.
        
        Returns:
            response (dict): The generated response from the LLM.
        """        
        # Prepare the input dictionary with the correct key
        input_dict = {"user_input": user_input}
        # Use the model to generate a response
        response = chain.invoke(input_dict)
        
        # This will return metadatasm. If you want response content only, use print(response.content) after instantiating LLM()
        return response
        