if  [ ! -n "$1" ]; then echo "dirname missing" && exit ; fi
python ../../seatable.py "$1"
