"""
Producer (or publisher) module to send messages to RabbitMQ.
"""

# Load packages.
import os
import pika


# Configure the RabbitMQ host and queue name.
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST")
QUEUE_NAME = "hello"

# Create a connection to the RabbitMQ server.
connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))

# Create a channel inside connection.
channel = connection.channel()

# If the queue does not exist, it will be created.
channel.queue_declare(queue=QUEUE_NAME)

# Publish a message to the queue.
channel.basic_publish(exchange="", routing_key=QUEUE_NAME, body="Hello World Message!")
print("[x] Sent Hello World Message!")

# Close the connection.
connection.close()
