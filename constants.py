import torchvision.transforms as transforms 

# Transformer
transform = transforms.Compose(
    [transforms.ToPILImage('RGB'),
     transforms.Resize([255, 255]),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Batch size
batch_size = 4

# Images classes
classes = ('Abra', 'Aerodactyl', 'Alakazam', 'Arbok', 'Arcanine', 'Articuno', 'Beedril', 'Bellsprout', 'Blastoise', 'Bulbasaur')