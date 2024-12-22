#!/usr/bin/env python
import os
import pika


# Configure the RabbitMQ host and queue name.
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST")


connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
channel = connection.channel()
channel.queue_declare(queue="hello")
channel.basic_publish(exchange="", routing_key="hello", body="Hello World!")

print(" [x] Sent 'Hello World!'")

connection.close()