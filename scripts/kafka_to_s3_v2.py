import sys
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from awsglue.dynamicframe import DynamicFrame

# Initialize Glue context and other parameters
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'KAFKA_TOPIC_NAME', 'S3_OUTPUT_PATH'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Kafka topic name and S3 output path
kafka_topic_name = args['KAFKA_TOPIC_NAME']
s3_output_path = args['S3_OUTPUT_PATH']

# Read data from Kafka
kafka_options = {
    "connectionName": "KafkaConnection",
    "topicName": kafka_topic_name
}

kafka_stream = glueContext.create_data_frame.from_catalog(
    database="market_db",
    table_name=kafka_topic_name,
    transformation_ctx="kafka_stream"
)

# Convert to a dynamic frame
dynamic_frame = DynamicFrame.fromDF(kafka_stream, glueContext, "dynamic_frame")

# Process data (optional, e.g., filter, transform, etc.)
# Uncomment the below example for a basic filter operation.
# dynamic_frame = dynamic_frame.filter(lambda x: x["price"] > 100)

# Write data to S3 as Parquet
s3_sink = glueContext.getSink(
    path=s3_output_path,
    connection_type="s3",
    updateBehavior="UPDATE_IN_DATABASE",
    partitionKeys=[],  # Add partition keys if needed
    enableUpdateCatalog=True
)
s3_sink.setFormat("glueparquet")
s3_sink.writeFrame(dynamic_frame)

# Commit the job
job.commit()
