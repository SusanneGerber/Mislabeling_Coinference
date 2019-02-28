classdef Timer < handle
   % Timer single node of Timers
   %
   % Lukas Pospisil (USI Lugano)
   % published under MIT Licence, 2017
   %
   
   properties(SetAccess = private)
      name   % name of timer (for futher reference)
      value  % sum of times between start() and stop()
   end
   
   properties(Access = private)
      tStart = 0      % tic start
      running = false % is the timer running or not? 
   end
   
   methods
      function timer = Timer(name)
         % constructor - create timer with name 
         timer.name = name;
         timer.value = 0;
      end
      
      function start(mytimer)
         % start to measure time 
         if mytimer.running % maybe this time is already running
            disp(['Warning: Timer ' mytimer.name ' is already running'])
         else
            mytimer.tStart = tic;
            mytimer.running = true;
         end
      end
      
      function stop(mytimer)
         % stop to measure time and store result
         if mytimer.running
            mytimer.value = mytimer.value + toc(mytimer.tStart);
            mytimer.running = false;
         else
             disp(['Warning: Timer ' mytimer.name ' is not running'])
         end
      end
      
      function out = ismyname(mytimer,new_name)
         % check if input is the name of this timer 
         if issame(new_name,mytimer.name)
            out = true;
         else
            out = false; 
         end
      end

      function out = isrunning(mytimer)
          % check if the timer is running or not
          out = mytimer.running;
      end

      function out = get_value(mytimer)
          % check if the timer is running or not
          out = mytimer.value;
      end      

      function out = get_name(mytimer)
          % check if the timer is running or not
          out = mytimer.name;
      end 
      
      function print(mytimer)
          % print values of this timer
          if mytimer.isrunning()   
              mymark = '*'; % if Timer is still running, then add this mark
          else
              mymark = '';
          end
          disp(['  - ' mytimer.name mymark ': ' num2str(mytimer.value) 's '])
      end
      
   end
end