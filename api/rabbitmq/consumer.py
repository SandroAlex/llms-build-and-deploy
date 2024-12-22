"""
Consumer script to consume messages from RabbitMQ.
"""

# Load packages.
import os
import sys
import pika


# Configure the RabbitMQ host and queue name.
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST")
QUEUE_NAME = "hello"


def main():
    """
    Main function to consume messages from RabbitMQ.
    """

    # Create a connection to the RabbitMQ server.
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))

    # Create a channel in the above connection.
    channel = connection.channel()

    # Create the queue. If the queue already exists, it will not be created.
    # Associate it with the channel.
    channel.queue_declare(queue=QUEUE_NAME)

    # Define a callback function to print the message received.
    def callback(channel, method, properties, body):
        print("[x] received %r" % body)

    # Associate the callback function with the message queue.
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=True)

    # Start consuming the messages.
    print(" [*] waiting for messages. To exit press Ctrl-C")
    channel.start_consuming()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        try:
            sys._exit(0)
        except SystemExit:
            os._exit(0)
