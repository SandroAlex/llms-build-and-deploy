# Load packages.
import os
import pika


# Configure the RabbitMQ host and queue name.
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST")


# Define a callback function to print the message received.
def callback(channel, method, properties, body):
    print("[x] received %r" % body)


# Create a connection.
connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))

# Create a channel in connection.
channel = connection.channel()

# Create the exchange (it will not actually create it if it already exists).
channel.exchange_declare(exchange="br_exchange", exchange_type="fanout")

# Create the temporary queue and associate it with the exchange exclusively.
result = channel.queue_declare(queue="", exclusive=True)
queue_name = result.method.queue

print(">>> Subscriber queue name: ", queue_name)

# Bind the queue to the exchange.
channel.queue_bind(exchange='br_exchange', queue=queue_name)

print("[*] Waiting for the messages ... ")

# Associate a callback function with the message queue.
channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)

channel.start_consuming()
