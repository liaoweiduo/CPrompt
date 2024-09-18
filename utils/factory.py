from models.cprompt import CPrompt

def get_model(model_name, args):
    name = model_name.lower()
    if "cprompt" in name:
        return CPrompt(args)
    else:
        assert 0
