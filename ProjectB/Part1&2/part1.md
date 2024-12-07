# Generate Dataset

`python ImageNet-1k/sort_imagenet.py ImageNet-1k/ train`

`python ImageNet-1k/sort_imagenet.py ImageNet-1k/ val`

# Run Vim

## Tiny model

python Vim/vim/main.py --eval --resume Vim/vim/weights/vim_t_midclstok_76p1acc.pth --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path ImageNet-1k --num_workers 0 --no-pin-mem

## Tiny+ model

python Vim/vim/main.py --eval --resume Vim/vim/weights/vim_t_midclstok_ft_78p3acc.pth --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path ImageNet-1k --num_workers 0 --no-pin-mem

## Small model

python Vim/vim/main.py --eval --resume Vim/vim/weights/vim_s_midclstok_80p5acc.pth --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path ImageNet-1k --num_workers 0 --no-pin-mem

## Small+ model

python Vim/vim/main.py --eval --resume Vim/vim/weights/vim_s_midclstok_ft_81p6acc.pth --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path ImageNet-1k --num_workers 0 --no-pin-mem

## Base model

python Vim/vim/main.py --eval --resume Vim/vim/weights/vim_b_midclstok_81p9acc.pth --model vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-path ImageNet-1k --num_workers 0 --no-pin-mem

# References

https://github.com/hustvl/Vim
