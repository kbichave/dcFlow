CUDA_VISIBLE_DEVICES=1,0 python main.py --exp_name=dcp_v2 --model=dcp  \
--emb_nn=dgcnn --pointer=transformer --head=pointnet \
--batch_size=4 --dataset=flyingthings3dflow  --test_batch_size=1 --model=dcflow \
--epochs=1000 --num_points=1024 
