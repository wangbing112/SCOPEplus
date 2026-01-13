import tensorflow as tf
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
import os

def convert_steps_to_epochs(input_file, output_file, total_steps=33150, total_epochs=200):
    """
    将TensorBoard日志文件的横轴从step转换为epoch
    
    Args:
        input_file: 原始events文件路径
        output_file: 输出events文件路径
        total_steps: 训练总step数
        total_epochs: 训练总epoch数
    """
    # 计算缩放比例
    scale_factor = total_epochs / total_steps
    
    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取原始事件文件
    print(f"读取原始文件: {input_file}")
    print(f"总steps: {total_steps}, 总epochs: {total_epochs}")
    print(f"缩放比例: {scale_factor:.6f}")
    
    # 创建写入器
    writer = tf.summary.create_file_writer(output_dir)
    
    event_count = 0
    scalar_count = 0
    
    # 读取并转换事件
    for record in tf_record.tf_record_iterator(input_file):
        event = event_pb2.Event()
        event.ParseFromString(record)
        event_count += 1
        
        # 如果是标量数据，转换step为epoch
        if event.HasField('summary'):
            for value in event.summary.value:
                if value.HasField('simple_value') or value.HasField('tensor'):
                    # 计算对应的epoch
                    original_step = event.step
                    epoch_step = int(original_step * scale_factor)
                    
                    # 使用writer写入转换后的数据
                    with writer.as_default():
                        if value.HasField('simple_value'):
                            tf.summary.scalar(value.tag, value.simple_value, step=epoch_step)
                            scalar_count += 1
                        elif value.HasField('tensor'):
                            # 处理tensor类型的数据
                            tensor_proto = value.tensor
                            tf.summary.scalar(value.tag, 
                                            tf.make_ndarray(tensor_proto).item(), 
                                            step=epoch_step)
                            scalar_count += 1
    
    writer.close()
    
    print(f"\n转换完成！")
    print(f"处理事件数: {event_count}")
    print(f"转换标量数: {scalar_count}")
    print(f"输出文件目录: {output_dir}")
    print(f"\n使用以下命令查看结果:")
    print(f"tensorboard --logdir {output_dir}")

if __name__ == "__main__":
    # 配置参数
    input_file = "./Jan08_19-39-59-scope-perovskites/events.out.tfevents.1767872399.autodl-container-ade2498877-806d79aa.238795.0"
    output_dir = "logs_by_epoch"  # 输出目录
    output_file = os.path.join(output_dir, "events.out.tfevents.epoch.scope-perovskites")
    
    total_steps = 11950  # 你的总step数
    total_epochs = 200   # 你的总epoch数
    
    # 执行转换
    convert_steps_to_epochs(input_file, output_file, total_steps, total_epochs)
