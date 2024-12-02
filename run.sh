#!/bin/bash

# Array of predefined prompts
prompts=(
    "A camel on a sand dune, with some cacti in the background."
    "A picture of a bike standing in a green room with its rear wheel leaning against a wooden table."
    "A picture of a man holding his surfboard and walking on the beach."
    "A picture of a hand holding a green apple."
    "A side profile shot of a road biker riding downhill."
    "Three sports cars chasing each other on the road, with one driver extending their hand out of the window, giving a victory sign."
    "A person sitting at a desk with a keyboard and monitor."
)

# Initialize counter
counter=1

# Generate images for each predefined prompt
for prompt in "${prompts[@]}"; do
    echo "Generating image for prompt: $prompt"
    PYTHONPATH=. python stable_diffusion.py --prompt "$prompt"
    mv output.png "output_${counter}.png"
    echo "Image saved as output_${counter}.png"
    ((counter++)) # Increment counter
done

echo "Image generation completed!"
