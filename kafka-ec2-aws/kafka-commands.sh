
### criar topico
kafka-topics.sh --create \
  --bootstrap-server localhost:2181 \
  --replication-factor 1 \
  --partitions 1 \
  --topic kafkamytopic