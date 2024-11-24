import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql import DataFrame, Row
import datetime
from awsglue import DynamicFrame

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
bucket_name = "lab-bg-02-s3bucket-pobm254onqaj"

# Script generated for node Apache Kafka
df_kafka = glueContext.create_data_frame.from_options(connection_type="kafka",connection_options={"bootstrap.servers": "privateip:9092", "connectionName": "KafkaConnection", "classification": "json", "startingOffsets": "earliest", "topicName": "market_tickers_data", "inferSchema": "true", "typeOfData": "kafka"}, transformation_ctx="df_kafka")

def processBatch(data_frame, batchId):
    if (data_frame.count() > 0):
        kafka_node = DynamicFrame.fromDF(glueContext.add_ingestion_time_columns(data_frame, "hour"), glueContext, "from_data_frame")
        # Script generated for node Amazon S3
        s3_node_path = "s3://bucket_name/output/"
        s3_node = glueContext.getSink(path=s3_node_path, connection_type="s3", updateBehavior="UPDATE_IN_DATABASE", partitionKeys=["ingest_year", "ingest_month", "ingest_day", "ingest_hour"], compression="snappy", enableUpdateCatalog=True, transformation_ctx="s3_node")
        s3_node.setCatalogInfo(catalogDatabase="market_db",catalogTableName="market-data")
        s3_node.setFormat("json")
        s3_node.writeFrame(kafka_node)
glueContext.forEachBatch(frame = df_kafka, batch_function = processBatch, options = {"windowSize": "100 seconds", "checkpointLocation": args["TempDir"] + "/" + args["JOB_NAME"] + "/checkpoint/"})
job.commit()