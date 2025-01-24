class Logger:
  def __init__(self, output_path=None):
    self.output_path = output_path

  def log(self, msg):
    print(msg)

    if self.output_path:
      with open(self.output_path, 'a') as f: f.write(msg + '\n')
