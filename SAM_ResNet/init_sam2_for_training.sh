cd ./SAM_ResNet
mkdir "./sam2_repo"
git clone https://github.com/facebookresearch/sam2.git ./sam2_repo
cp -r ./configs/* ./sam2_repo/sam2/configs
cd ./sam2_repo/checkpoints && ./download_ckpts.sh