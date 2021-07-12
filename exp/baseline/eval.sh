REL_PATH=../../
DIR_NAME="${PWD##*/}"

PYTHONPATH=${PYTHONPATH}:${REL_PATH} GLOG_vmodule=MemcachedClient=-1 \
spring.submit run --gpu -n1 \
--ntasks-per-node=1 \
--cpus-per-task=5 \
--job-name "eval-${DIR_NAME}" "python -u -m eval --ckpt_path=\"$1\" --cfg=cfg.yaml"


