import os
import pika


# Configure the RabbitMQ host and queue name.
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST")

# Create a connection to the RabbitMQ server.
connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))

# Create a channel inside connection.
channel = connection.channel()

# Create the exchange (it will not actually create it if it already exists).
channel.exchange_declare(exchange="br_exchange", exchange_type="fanout")

# Publish a number of messages to the exchange.
for i in range(10):
    message = f"Hello World Message {i}!"
    channel.basic_publish(exchange="br_exchange", routing_key="", body=message)
    print(f"[x] Sent {message}")

# Delete the exchange.
channel.exchange_delete(exchange="br_exchange", if_unused=False)

# Close the connection.
connection.close()
