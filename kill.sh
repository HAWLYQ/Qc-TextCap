ps -ef|grep pythia|grep -v grep|awk '{print "kill -9 "$2}'
ps -ef|grep pythia|grep -v grep|awk '{print "kill -9 "$2}'|sh
