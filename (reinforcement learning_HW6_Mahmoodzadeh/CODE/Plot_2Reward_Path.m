% Water Maze Task
% Implementation in MATLAB

% Parameters
mapSize = 15;  % Size of the map
numTrials = 500;  % Number of trials
eta = 0.3;  % Learning rate
gamma = 0.9;  % Discount factor

% Initialize the map
map = zeros(mapSize, mapSize);  % Create a map of zeros
target = [3, 12];  % Target location (black)
cat = [8, 8];  % Cat location (red)
additionalReward = [14, 1];  % Additional reward location (green)

% Initialize the state values
V = zeros(mapSize, mapSize);  % Initialize all state values to zero
V(target(1), target(2)) = 10;  % Set the target state value to reward
V(cat(1), cat(2)) = -10;  % Set the cat state value to punishment
V(additionalReward(1), additionalReward(2)) = 5;  % Set the additional reward state value

% Run the trials
figure;

% Define the specific trials to plot
specificTrials = [10, 20, 30];

for trial = 1:numTrials
    % Randomly select the starting position
    start = randi([2, mapSize-1], 1, 2);  % Exclude the edges
    
    % Simulate agent's movement until reaching the target or cat or additional reward
    currentState = start;
    path = currentState;
    
    while ~isequal(currentState, target) && ~isequal(currentState, cat) && ~isequal(currentState, additionalReward)
        % Get the available actions
        actions = getAvailableActions(currentState, mapSize);
        
        % Choose the action with the highest value (or randomly among equal values)
        nextAction = chooseAction(currentState, actions, V);
        
        % Take the selected action and transition to the next state
        nextState = transitionState(currentState, nextAction);
        
        % Move to the next state
        currentState = nextState;
        path = [path; currentState];
    end
    
    % Plot the path for specific trials
    if ismember(trial, specificTrials)
        subplot(length(specificTrials), 1, find(specificTrials == trial));
        plot(path(:, 2), path(:, 1), 'Color', 'k');
        hold on;
        
        % Plot the start point as a yellow point
        plot(path(1, 2), path(1, 1), 'o', 'MarkerSize', 6, 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k');
        
        % Plot the Reward and Punishment locations
        plot(target(2), target(1), 'sk', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
        plot(cat(2), cat(1), 'sr', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
        plot(additionalReward(2), additionalReward(1), 'sg', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
        
        xlim([1 mapSize]);
        ylim([1 mapSize]);
        title(sprintf('Agent Path (Trial %d)', trial));
        xlabel('Column');
        ylabel('Row');
    end
    
    % Pause for a brief moment to visualize each trial
    pause(0.5);
end

% Function to get available actions for a given state
function actions = getAvailableActions(state, mapSize)
    actions = [];
    if state(1) > 1
        actions = [actions; -1, 0];  % Up
    end
    if state(1) < mapSize
        actions = [actions; 1, 0];  % Down
    end
    if state(2) > 1
        actions = [actions; 0, -1];  % Left
    end
    if state(2) < mapSize
        actions = [actions; 0, 1];  % Right
    end
end

% Function to choose the action with the highest value (or randomly among equal values)
function action = chooseAction(state, actions, V)
    actionValues = zeros(size(actions, 1), 1);
    
    for i = 1:size(actions, 1)
        nextPos = state + actions(i, :);
        actionValues(i) = V(nextPos(1), nextPos(2));
    end
    
    [~, maxIdx] = max(actionValues);
    maxValuesIdx = find(actionValues == actionValues(maxIdx));
    action = actions(maxValuesIdx(randi(numel(maxValuesIdx))), :);
end

% Function to transition from the current state to the next state
function nextState = transitionState(currentState, action)
    nextState = currentState + action;
end
