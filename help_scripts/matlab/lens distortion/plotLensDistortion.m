function plotLensDistortion(distortionX,distortionY,parameters,parameterValues)
% distortionX is the expression describing the distorted x coordinate
% distortionY is the expression describing the distorted y coordinate
% k1 and k2 are the radial distortion coefficients 
% p1 and p2 are the tangential distortion coefficients 

syms x y

% This is the grid spacing over the image
spacing = 0.2;

% Inspect and parametrically substitute in the values for k_1 k_2 p_1 p_2
distortionX = subs(distortionX,parameters,parameterValues);
distortionY = subs(distortionY,parameters,parameterValues);

% Loop over the grid
for x_i = -1:spacing:1
    for y_j = -1:spacing:1
        
        % Compute the distorted location 
        xout = subs(distortionX, {x,y}, {x_i,y_j});
        yout = subs(distortionY, {x,y}, {x_i,y_j});
        
        % Plot the original point
        plot(x_i,y_j, 'o', 'Color', [1.0, 0.0, 0.0])
        hold on
       
        % Draw the distortion direction with Quiver
        p1 = [x_i,y_j];                     % First Point
        p2 = [xout,yout];                   % Second Point
        dp = p2-p1;                         % Difference
        quiver(p1(1),p1(2),dp(1),dp(2),'AutoScale','off','MaxHeadSize',1,'Color',[0 0 1])
        
    end
end
hold off
grid on

end