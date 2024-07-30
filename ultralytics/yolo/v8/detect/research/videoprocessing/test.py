import sys
import os

# Add the deep_sort_pytorch directory to the sys.path
current_file_path = os.path.abspath(os.path.dirname(__file__))
deep_sort_path = os.path.abspath(os.path.join(current_file_path, '..', 'deep_sort_pytorch'))
if deep_sort_path not in sys.path:
    sys.path.append(deep_sort_path)

# Now try importing the deep_sort_pytorch modules
try:
    from deep_sort_pytorch.utils.parser import get_config
    from deep_sort_pytorch.deep_sort import DeepSort
    print("Imports successful!")
except ModuleNotFoundError as e:
    print(f"Error: {e}")