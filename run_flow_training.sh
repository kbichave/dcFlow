CUDA_VISIBLE_DEVICES=0,1 python main.py --exp_name=dcp_v2 --model=dcp  \
--emb_nn=dgcnn --pointer=transformer --head=pointnet \
--batch_size=4 --dataset=kitti2015flow  --test_batch_size=1 --model=dcflow \
--epochs=1000 --num_points=1024 --onlytrain
