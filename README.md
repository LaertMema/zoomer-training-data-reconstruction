# Zoomer Challenge (Training Data Reconstruction)

This project documents my solution to the Zoomer Challenge from the PINE 26 seminar. The task was to analyze an authentication executable, recover its embedded machine learning model, and reconstruct 18 hidden training images. Submissions were evaluated using mean cosine similarity across the hidden samples.

The approach combined reverse engineering and natural image retrieval:

- Recover the preprocessing pipeline and model weights, then reconstruct the ResNet-18 classifier with a binary output head in PyTorch.
- Retrieve candidate images from COCO, Visual Genome, Places365, and Open Images, using classifier scores and CLIP semantic embeddings to refine the search. DINOv2 was evaluated as a comparison.
- Improve the final 18-image set through anchor ablation and individual candidate swaps, evaluated by the challenge server.

The final submission achieved a mean cosine similarity score of **0.6657**.

Repository contents:

- [`source/`](source/): model reconstruction, retrieval, selection, and ablation scripts, along with prompts and experiment results.
- [`source/Final image candidates/`](source/Final%20image%20candidates/): the final 18 candidate images, selection data, and a contact sheet.
- [`source/Reverse Engineering Input Probes/`](source/Reverse%20Engineering%20Input%20Probes/): probe images used to investigate the executable.
- [`Report_for_Zoomer.pdf`](Report_for_Zoomer.pdf): the full challenge description, methodology, and results.

The scripts retain their original experiment paths and require external model weights, datasets, helper modules, and challenge infrastructure to reproduce the experiments.
