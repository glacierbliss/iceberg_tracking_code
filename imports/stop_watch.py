import datetime as dt

#def days_hours_minutes_seconds(td):
#    return td.days, td.seconds // 3600, (td.seconds % 3600) // 60, td.seconds % 60

class stop_watch:
    
    def __init__(self, func = dt.datetime.now):
        self.elapsed = 0.0
        self._func = func
        self._start = None
        self._intermediate = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')
        print('Start processing: ' + dt.datetime.now().strftime('%I:%M:%S'))
        self.elapsed = 0.0
        self._start = self._func()
        self._intermediate = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()
        self.elapsed = end - self._start
        self._start = None
        self._intermediate = None

    def print_elapsed_time(self):

        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()

        self.elapsed = end - self._start

        print('End processing: ' + dt.datetime.now().strftime('%I:%M:%S'))
        
        days = self.elapsed.days 
        hours = self.elapsed.seconds // 3600 
        minutes = (self.elapsed.seconds % 3600) // 60 
        seconds = self.elapsed.seconds % 60

        if days < 1:

            if hours < 1:
                
                if minutes < 1:
                    string = 'Elapsed time: %d s' % (seconds)
                else:
                    string = 'Elapsed time: %d min, and %d s' % (minutes, seconds)
                    
            else:
                string = 'Elapsed time: %d h, %d min, and %d s' % (hours, minutes, seconds)
                
        else:
            string = 'Elapsed time: %d d, %d h, %d min, and %d s' % (days, hours, minutes, seconds)
            
            
        print(string)

    def print_intermediate_time(self):

        if self._start is None:
            raise RuntimeError('Not started')
        end = self._func()

        self.elapsed = end - self._intermediate

        days = self.elapsed.days 
        hours = self.elapsed.seconds // 3600 
        minutes = (self.elapsed.seconds % 3600) // 60 
        seconds = self.elapsed.seconds % 60

        if days < 1:

            if hours < 1:
                
                if minutes < 1:
                    string = 'Elapsed time: %d s' % (seconds)
                else:
                    string = 'Elapsed time: %d min, and %d s' % (minutes, seconds)
                    
            else:
                string = 'Elapsed time: %d h, %d min, and %d s' % (hours, minutes, seconds)
                
        else:
            string = 'Elapsed time: %d d, %d h, %d min, and %d s' % (days, hours, minutes, seconds)
            
        print(string)
        self._intermediate = self._func()

    def reset(self):
        self.elapsed = 0.0
        self._start = self._func()

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()