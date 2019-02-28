classdef Timers < handle
   % Timers smart tic-toc measurements
   %  call with:
   %   timers = Timers();
   %   timers.start('my unique name');
   %    .. computation to be measured ..
   %   timers.stop('my unique name');
   %   timers.print();
   %
   % Lukas Pospisil (USI Lugano)
   % published under MIT Licence, 2017
   %
   
    properties(Access = private)
      list_of_timers  % here store called timers
   end
   
   methods
      function timers = Timers()
         % constructor - initialize list of timers
         timers.list_of_timers = cell(0);
      end
      
      function start(mytimers,name)
         % start timer identified by name  

         % go through list of timers and find the appropriate one
         found = false;
         for i=1:length(mytimers.list_of_timers)
            if mytimers.list_of_timers{i}.ismyname(name)
                mytimers.list_of_timers{i}.start();
                found = true;
            end
         end
      
         if ~found
            % this is a new timer, create it and run
            mytimers.list_of_timers{length(mytimers.list_of_timers) + 1} = Timer(name);
            mytimers.list_of_timers{length(mytimers.list_of_timers)}.start();
         end
      end
      
      function stop(mytimers,name)
         % go through list of timers and find the appropriate one
         found = false;
         for i=1:length(mytimers.list_of_timers)
            if mytimers.list_of_timers{i}.ismyname(name)
                mytimers.list_of_timers{i}.stop();
                found = true;
            end
         end
      
         if ~found
            disp(['Warning: Timer ' name ' is not running'])
            % do nothing
         end
      end

      function stop_all(mytimers)
         % go through list of timers and stop all timers
         for i=1:length(mytimers.list_of_timers)
             if mytimers.list_of_timers{i}.isrunning()
                 mytimers.list_of_timers{i}.stop();
             end
         end
      end
      
      function print(mytimers)
         % print values of all stored timers
         disp('TIMERS:')
         for i=1:length(mytimers.list_of_timers)
            mytimers.list_of_timers{i}.print(); 
         end
      end
      
      function out = get_value(mytimers,name)
         % start timer identified by name  

         % go through list of timers and find the appropriate one
         found = false;
         found_idx = -1;
         for i=1:length(mytimers.list_of_timers)
            if mytimers.list_of_timers{i}.ismyname(name)
                found = true;
                found_idx = i;
            end
         end
      
         if ~found
            disp(['Warning: Timer ' name ' does not exist'])
            out = 0;
         else
            out = mytimers.list_of_timers{found_idx}.get_value();
         end
      end

      function out = isrunning(mytimers,name)
         % check if the timer is running or not

         % go through list of timers and find the appropriate one
         found = false;
         found_idx = -1;
         for i=1:length(mytimers.list_of_timers)
            if mytimers.list_of_timers{i}.ismyname(name)
                found = true;
                found_idx = i;
            end
         end
      
         if ~found
            disp(['Warning: Timer ' name ' does not exist'])
            out = false;
         else
            out = mytimers.list_of_timers{found_idx}.isrunning();
         end
      end      
      
      function out = get_values(mytimers)
         % return values of timers as vector
         
         out = zeros(1,length(mytimers.list_of_timers));
         for i=1:length(mytimers.list_of_timers)
             out(i) = mytimers.list_of_timers{i}.get_value();
         end
      end
      
      function out = get_names(mytimers)
         % return names of timers as cell
         
         out = cell(1,length(mytimers.list_of_timers));
         for i=1:length(mytimers.list_of_timers)
             out{i} = mytimers.list_of_timers{i}.get_name();
         end
      end
      
      function out = get_timer(mytimers,name)
         % return timer identified by name
         
         % go through list of timers and find the appropriate one
         found = false;
         found_idx = -1;
         for i=1:length(mytimers.list_of_timers)
            if mytimers.list_of_timers{i}.ismyname(name)
                found = true;
                found_idx = i;
            end
         end
      
         if ~found
            disp(['Warning: Timer ' name ' does not exist'])
            out = [];
         else
            out = mytimers.list_of_timers{found_idx};
         end
      end
      
   end
end