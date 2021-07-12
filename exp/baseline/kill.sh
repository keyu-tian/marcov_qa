_IFS="$IFS"
IFS=$'\n'

function kills() {
  name=$1
  for pid in $(ps | grep "$name" | grep -v grep | awk '{print $1}'); do
    kill $pid
  done
}

kills python
kills tensorboard
rm -f "./*.terminate"

IFS="$_IFS"

