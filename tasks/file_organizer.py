import os
import shutil
import datetime

def run_file_organizer(target_folder="C:\\Users\\Harshit\\Downloads"):
    # Define common file type categories
    file_types = {
        "Images": [".jpg", ".jpeg", ".png", ".gif"],
        "Documents": [".pdf", ".docx", ".txt", ".pptx"],
        "Videos": [".mp4", ".mkv", ".avi"],
        "Music": [".mp3", ".wav"],
        "Archives": [".zip", ".rar", ".7z"],
        "Code": [".py", ".java", ".cpp", ".c", ".js", ".html", ".css"]
    }

    if not os.path.exists(target_folder):
        return "Target folder not found!"

    # Create log directory if missing
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    moved_count = 0
    for file in os.listdir(target_folder):
        file_path = os.path.join(target_folder, file)

        if os.path.isfile(file_path):
            file_ext = os.path.splitext(file)[1].lower()
            for folder, extensions in file_types.items():
                if file_ext in extensions:
                    folder_path = os.path.join(target_folder, folder)
                    os.makedirs(folder_path, exist_ok=True)
                    shutil.move(file_path, os.path.join(folder_path, file))
                    moved_count += 1
                    break

    log_message = f"[{timestamp}] File Organizer moved {moved_count} files in {target_folder}\n"
    with open("logs/task_log.txt", "a") as f:
        f.write(log_message)

    return f"File Organizer Task Complete! Moved {moved_count} files."
