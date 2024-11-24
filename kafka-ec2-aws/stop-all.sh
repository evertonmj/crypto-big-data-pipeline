!#bash

KAFKA_HOME=/home/ec2-user/kafka_2.12-3.8.0
echo "Parando kafka...."
$KAFKA_HOME/bin/kafka-server-stop.sh
echo "Parando zookeeper...."
$KAFKA_HOME/bin/zookeeper-server-stop.sh