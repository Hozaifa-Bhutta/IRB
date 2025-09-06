import json, os



def write_to_jsonl(data, filename):
    with open(filename, "a") as f:
        for line in data:
            json.dump(line, f)
            f.write("\n")


def write_to_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def read_json_or_jsonl(filename):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except Exception:
        with open(filename, "r") as f:
            data = []
            for line in f:
                data.append(json.loads(line))

    return data

def maybe_create_folder(folder_path):
    """
    Creates a folder at the specified path if it doesn't already exist.

    Args:
        folder_path (str): The path of the folder to create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")