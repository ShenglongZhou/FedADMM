function  PlotObj(obj,k0)
    figure('Renderer', 'painters', 'Position',[1100 400 370 320]);
    axes('Position', [0.16 0.14 0.82 0.8] ); 
    y  = obj(1:k0:end);
    h1 = plot(1:length(y),y); hold on
    grid on
    h1.LineWidth  = 1.5;      
    h1.LineStyle  = '-';   
    h1.Color = '#3caea3';  
    axis([1 length(y) min(obj) max(obj)])
    xlabel('CR'); ylabel('Objective'); 
    legend('FedADMM')
end
