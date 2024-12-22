#!/usr/bin/env python
import os
import sys
import pika


# Configure the RabbitMQ host and queue name.
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST")


connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
channel = connection.channel()
channel.queue_declare(queue="hello")
message = " ".join(sys.argv[1:]) or "Hello World!"
channel.basic_publish(exchange="", routing_key="hello", body=message)

print(f" [x] Sent {message}")