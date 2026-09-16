from gradio_client import Client
client = Client("nd1490/ratatouille-inference")
print(client.view_api(return_format="dict"))
