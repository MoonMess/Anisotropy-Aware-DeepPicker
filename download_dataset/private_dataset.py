from cryoet_data_portal import Client, Dataset

client = Client()
dest_path = "../../storage/data/czi/private_dataset/"

dataset = Dataset.get_by_id(client, 10446)
dataset.download_everything(dest_path=dest_path)
