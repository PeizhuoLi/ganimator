for i in {1..30}
do
  python demo.py --save_path=./pre-trained/Eight/
  mv ./pre-trained/Eight/bvh ./pre-trained/Eight/bvh_$i
done

