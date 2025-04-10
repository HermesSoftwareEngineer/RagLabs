def act_class_to_dict_schema(act_class):
    """
    Converts a Pydantic class with a Union field and discriminator to a dictionary schema
    suitable for use with language models for structured output.

    Args:
        act_class: The Pydantic class to convert. It should have an 'action' field
                   that is a Union of other Pydantic classes, and a 'discriminator'
                   defined on the Union.

    Returns:
        A dictionary representing the schema.
    """

    # Access the 'action' field and its discriminator
    action_field = act_class.__fields__['action']
    discriminator_field = action_field.discriminator  # Use the correct attribute
    union_types = action_field.annotation.__args__  # Access the Union types

    properties = {
        "action": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": [t.__fields__[discriminator_field].default for t in union_types]},
            },
            "required": ["type"]
        }
    }

    # Add properties from each of the Union types
    for union_type in union_types:
        for field_name, field_info in union_type.__fields__.items():
            if field_name != discriminator_field:
                # Map Python types to JSON schema types
                python_type = field_info.annotation
                json_type = "string" if python_type == str else "number" if python_type in [int, float] else "boolean" if python_type == bool else "object"

                properties["action"]["properties"][field_name] = {
                    "type": json_type,  # Use mapped JSON type
                    "description": field_info.description  # Access description directly
                }
                properties["action"]["required"] = properties["action"].get("required", []) + [field_name]

    schema = {
        "name": "Act",
        "description": "Escolher qual a melhor forma para responder",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": ["action"]
        }
    }
    return schema