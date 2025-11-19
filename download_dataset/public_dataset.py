from cryoet_data_portal import Client, Dataset

client = Client()
dest_path = "../../storage/data/czi/public_dataset/"

dataset = Dataset.get_by_id(client, 10445)
dataset.download_everything(dest_path=dest_path)
