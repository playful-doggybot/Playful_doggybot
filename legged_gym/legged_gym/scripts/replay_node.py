import argparse
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String
import rosbag2_py

import os.path as osp
import time

from play_node import *

class replayNode(PlayNode):
    def __init__(self, args):
        super().__init__(args)


def read_messages(input_bag: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_bag, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    topic_types = reader.get_all_topics_and_types()

    def typename(topic_name):
        for topic_type in topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"topic {topic_name} not in bag")

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg_type = get_message(typename(topic))
        msg = deserialize_message(data, msg_type)
        yield topic, msg, timestamp
    del reader

def main():
    args = parser.parse_args()
    input_bag = osp.join("/home/duanxin/Documents/Legged_robot/doggy_bots/legged_gym/logs/go2_mix_cmd", args.load_run)
    input_bag = osp.join(input_bag, args.input)

    last_timestamp = -1
    for topic, msg, timestamp in read_messages(input_bag):
        if topic == "/sim_logs/obs":
            print(f"{topic} [{timestamp}]: '{msg.data}'")
            if last_timestamp >= 0:
                time.sleep((timestamp - last_timestamp)*1.e-9)
            last_timestamp = timestamp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load_run", help="input checkpoint path (folder or filepath) to read from"
    )
    parser.add_argument(
        "--input", help="input bag path (folder or filepath) to read from"
    )