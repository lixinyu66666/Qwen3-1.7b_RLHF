def format_prompt(instruction, response):
    """
    Formats the instruction and response into a structured prompt.
    
    Args:
        instruction (str): The instruction part of the prompt.
        response (str): The response part of the prompt.
        
    Returns:
        str: A formatted string combining the instruction and response.
    """
    return f"""### Instruction:\n{instruction}\n\n### Response:\n{response}"""
