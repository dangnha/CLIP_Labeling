import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Database
DATABASE_PATH = BASE_DIR / "image_db.sqlite"

# File uploads
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Model paths
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
CLIP_MODEL = "ViT-B/32"

# App config
SECRET_KEY = "fsudyfgasjfbdsahfga12322312"

# Add default categories for image classification
DEFAULT_CATEGORIES = [
    "person", "animal", "vehicle", "building", "nature", "food", "indoor", "outdoor", "day", "night",
    "text", "art", "technology", "sports", "furniture", "clothing", "accessory", "electronic", "appliance", "tool",
    "plant", "flower", "tree", "landscape", "mountain", "ocean", "river", "lake", "sky", "cloud",
    "sun", "moon", "star", "fire", "water", "ice", "snow", "rain", "storm", "fog",
    "desert", "forest", "jungle", "grassland", "cave", "island", "beach", "coast", "valley", "hill",
    "road", "bridge", "railway", "airport", "harbor", "factory", "farm", "market", "store", "school",
    "hospital", "house", "apartment", "skyscraper", "castle", "temple", "church", "mosque", "monument", "tower",
    "statue", "painting", "drawing", "sculpture", "graffiti", "photograph", "film", "book", "newspaper", "magazine",
    "computer", "smartphone", "tablet", "television", "camera", "radio", "drone", "robot", "microscope", "telescope",
    "bicycle", "motorcycle", "car", "truck", "bus", "train", "airplane", "boat", "submarine", "helicopter",
    "rocket", "skateboard", "rollerblade", "wheelchair", "scooter", "cart", "carriage", "canoe", "yacht", "surfboard",
    "soccer", "basketball", "tennis", "baseball", "golf", "swimming", "running", "cycling", "skiing", "snowboarding",
    "fishing", "hunting", "wrestling", "boxing", "karate", "judo", "gymnastics", "horse riding", "archery", "fencing",
    "football", "rugby", "hockey", "volleyball", "badminton", "table tennis", "skating", "ballet", "yoga", "aerobics",
    "hamburger", "pizza", "pasta", "bread", "cake", "ice cream", "chocolate", "coffee", "tea", "wine",
    "beer", "soda", "juice", "salad", "sandwich", "fruit", "vegetable", "cheese", "meat", "fish",
    "seafood", "soup", "noodle", "sushi", "spice", "snack", "dessert", "breakfast", "lunch", "dinner",
    "dog", "cat", "horse", "cow", "sheep", "goat", "pig", "chicken", "duck", "goose",
    "rabbit", "deer", "elephant", "lion", "tiger", "bear", "wolf", "fox", "monkey", "panda",
    "zebra", "giraffe", "kangaroo", "penguin", "whale", "dolphin", "shark", "octopus", "crocodile", "turtle",
    "snake", "frog", "butterfly", "bee", "ant", "spider", "bat", "eagle", "owl", "parrot",
    "pigeon", "peacock", "swan", "flamingo", "rooster", "hawk", "vulture", "seagull", "pelican", "woodpecker",
    "suit", "dress", "jeans", "t-shirt", "jacket", "coat", "hat", "cap", "scarf", "glove",
    "sock", "shoe", "boot", "sandal", "watch", "bracelet", "necklace", "earring", "ring", "belt",
    "glasses", "sunglasses", "backpack", "handbag", "wallet", "umbrella", "tie", "helmet", "mask", "glove",
    "carpet", "curtain", "bed", "sofa", "chair", "table", "desk", "wardrobe", "shelf", "mirror",
    "lamp", "clock", "candle", "painting", "frame", "vase", "flower pot", "bathtub", "toilet", "sink",
    "shampoo", "soap", "toothbrush", "towel", "razor", "comb", "hairdryer", "perfume", "makeup", "lotion",
    "phone", "keyboard", "mouse", "monitor", "printer", "scanner", "router", "microphone", "speaker", "headphones",
    "guitar", "piano", "violin", "drum", "trumpet", "flute", "saxophone", "harmonica", "accordion", "harp",
    "book", "notebook", "paper", "pen", "pencil", "marker", "eraser", "ruler", "calculator", "stapler",
    "knife", "fork", "spoon", "plate", "bowl", "cup", "mug", "bottle", "thermos", "kettle",
    "microwave", "oven", "stove", "refrigerator", "freezer", "blender", "toaster", "coffee maker", "rice cooker", "dishwasher",
    "hammer", "wrench", "screwdriver", "pliers", "saw", "drill", "tape measure", "level", "ladder", "paintbrush",
    "tent", "sleeping bag", "campfire", "binoculars", "compass", "map", "flashlight", "lifeboat", "oxygen tank", "scuba gear",
    "currency", "coin", "bill", "credit card", "passport", "ticket", "badge", "medal", "trophy", "certificate",
    "alien", "robot", "spaceship", "portal", "time machine", "magic wand", "spell book", "treasure", "pirate", "knight"
]
