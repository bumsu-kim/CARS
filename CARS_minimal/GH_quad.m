%% Gauss - Hermite Quadratures
% valid inputs: 3, 4, 5, 6, 10
function [x, w] = GH_quad(m)
    if m==3
        x = [ -1.224744871391589049099;
            0;
            1.224744871391589049099];
        w = [0.295408975150919337883;
            1.181635900603677351532;
            0.295408975150919337883];
    elseif m==4
        x = [-1.650680123885784555883;
            -0.5246476232752903178841;
            0.5246476232752903178841;
            1.650680123885784555883];
        w = [0.081312835447245177143;
            0.8049140900055128365061;
            0.8049140900055128365061;
            0.08131283544724517714303];
    elseif m==5
        x = [-2.020182870456085632929;
            -0.9585724646138185071128;
            0;
            0.9585724646138185071128;
            2.020182870456085632929];
        w = [0.01995324205904591320774;
            0.3936193231522411598285;
            0.9453087204829418812257;
            0.393619323152241159828;
            0.01995324205904591320774];
    elseif m==6
        x = [-2.350604973674492222834;
            -1.335849074013696949715;
            -0.4360774119276165086792;
            0.436077411927616508679;
            1.335849074013696949715;
            2.350604973674492222834];
        w = [0.004530009905508845640857;
            0.1570673203228566439163;
            0.7246295952243925240919;
            0.724629595224392524092;
            0.1570673203228566439163;
            0.004530009905508845640857];
    elseif m==7
        x = [-2.651961356835233492447;
            -1.673551628767471445032;
            -0.8162878828589646630387;
            0;
            0.8162878828589646630387;
            1.673551628767471445032;
            2.651961356835233492447];
        w = [9.71781245099519154149E-4;
            0.05451558281912703059218;
            0.4256072526101278005203;
            0.810264617556807326765;
            0.4256072526101278005203;
            0.0545155828191270305922;
            9.71781245099519154149E-4];
    elseif m==10
        x = [-3.436159118837737603327
            -2.532731674232789796409
            -1.756683649299881773451
            -1.036610829789513654178
            -0.3429013272237046087892
            0.3429013272237046087892
            1.036610829789513654178
            1.756683649299881773451
            2.532731674232789796409
            3.436159118837737603327];
        w = [7.64043285523262062916E-6
            0.001343645746781232692202
            0.0338743944554810631362
            0.2401386110823146864165
            0.6108626337353257987836
            0.6108626337353257987836
            0.2401386110823146864165
            0.03387439445548106313616
            0.001343645746781232692202
            7.64043285523262062916E-6];
    end
end