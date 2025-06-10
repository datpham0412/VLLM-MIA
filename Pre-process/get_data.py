from datasets import load_dataset, load_from_disk

# Local path to save the dataset
save_path_img_Flickr = "/fred/oz402/aho/VLLM-MIA/Data/img_Flickr"
save_path_img_dalle = "/fred/oz402/aho/VLLM-MIA/Data/img_dalle"
save_path_text = "/fred/oz402/aho/VLLM-MIA/Data/llava_v15_gpt_text"

# -------------------------------------------------------------------
                            # image #
# # Flirckr set
# img_subset = "img_Flickr"
# image_dataset = load_dataset("JaineLi/VL-MIA-image", img_subset, split='train')
# image_dataset.save_to_disk(save_path_img_Flickr)

# # DALL-E set
# img_subset = "img_dalle"
# image_dataset = load_dataset("JaineLi/VL-MIA-image", img_subset, split='train')
# image_dataset.save_to_disk(save_path_img_dalle)

# -------------------------------------------------------------------
                            # text #
# text_len = 64 # 16,32,64
# text_subset = "llava_v15_gpt_text" # or minigpt4_stage2_text
# text_dataset = load_dataset("JaineLi/VL-MIA-text", text_subset, split=f"length_{text_len}")
# text_dataset.save_to_disk(save_path_text)

# --------------------------------------------------------------------
# check data #

dataset = load_from_disk(save_path_img_dalle)
print(dataset)
print(dataset[5])